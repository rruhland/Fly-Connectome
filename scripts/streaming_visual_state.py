"""Opt-in frame/event visual state with strictly observed delayed local credit."""

from collections import deque

import torch

from frame_observation_state import FrameObservationState, FrameObservationTracker
from generic_motion_probe import local_motion_map
from local_motion_dynamics import LocalMotionDynamics
from local_observation_model import LocalObservationModel
from local_metric_association import LocalMetricAssociation
from contextual_motion_dynamics import ContextualMotionDynamics
from associative_patch_state import PatchAssociation


class EndpointCalibration:
    """Observable reidentification, never a claim about hidden existence."""

    def __init__(self):
        self.outcomes = deque(maxlen=256)
        self.residuals = deque(maxlen=256)

    def observe(self, reidentified, residual):
        self.outcomes.append(bool(reidentified))
        if reidentified:
            self.residuals.append(float(residual))

    def summary(self):
        return dict(reidentification_probability=(1+sum(self.outcomes))/(2+len(self.outcomes)),
                    conditional_radius_90=(float(torch.quantile(torch.tensor(list(self.residuals)), .9))
                                           if self.residuals else None),
                    samples=len(self.outcomes))


class StreamingVisualState:
    def __init__(self, *, height=32, width=64, observer=None, memory=None, dynamics=None):
        self.observer = observer if observer is not None else LocalObservationModel(height=height, width=width)
        self.state = FrameObservationState(height=height, width=width, memory=memory)
        self.dynamics = (dynamics if dynamics is not None else
                         {h: LocalMotionDynamics() for h in (1, 4, 8)})
        self.calibration = {h: EndpointCalibration() for h in self.dynamics}
        self.reset_state()

    def reset_state(self):
        self.observer.reset_state()
        self.state.reset_state()
        self.trace = torch.zeros(2, self.observer.height, self.observer.width)
        self.histories = {}
        self.pending = {}
        self.forecast_cache = {}
        self.next_id = 0

    def compact(self):
        tracker = self.state.tracker
        keep = [i for i, slot in enumerate(tracker.slots)
                if slot['hypotheses'] and tracker.frame-slot['last_seen'] <= 64]
        tracker.slots = [tracker.slots[i] for i in keep]
        self.state.keys = {new: self.state.keys[old] for new, old in enumerate(keep)
                           if old in self.state.keys}
        self.state.confidence = {new: self.state.confidence[old] for new, old in enumerate(keep)
                                 if old in self.state.confidence}
        active = {slot['public_id'] for slot in tracker.slots}
        self.histories = {key: value for key, value in self.histories.items() if key in active}

    def frame_contrast(self, visible):
        return visible-visible.median()

    @torch.no_grad()
    def step(self, events, visible_frame, *, learn=False):
        available = visible_frame is not None
        filtered, _ = self.observer.step(events)
        if available:
            visible = visible_frame.float()
            self.observer.contrast.copy_(self.frame_contrast(visible))
        motion = local_motion_map(self.trace, filtered).clamp(-1., 1.)
        self.trace.mul_(.875).add_(filtered)
        contrast = self.observer.contrast
        sensory = torch.cat((filtered, self.observer.fast, self.observer.slow,
                              contrast.clamp(min=0.)[None], (-contrast).clamp(min=0.)[None],
                              motion, self.trace, events))
        field = self.state.step(sensory, learn=learn, observation_available=available)
        tracker = self.state.tracker
        for slot in tracker.slots:
            if 'public_id' not in slot:
                slot['public_id'] = self.next_id
                self.next_id += 1
        entities = []
        observed = {}
        for entity in self.state.entities:
            slot = tracker.slots[entity['id']]
            identity = slot['public_id']
            age = tracker.frame-slot['last_seen']
            row = dict(id=identity, position=entity['position'].clone(),
                       observed=age == 0, observation_age=age,
                       association_strength=entity['strength'])
            entities.append(row)
            if age == 0:
                observed[identity] = row['position']
        # Current sensor endpoints settle only forecasts made in the past.
        for item in self.pending.pop(tracker.frame, []):
            if not learn or not available:
                continue
            identity, horizon, history, origin, prediction, credit = item
            endpoint = observed.get(identity)
            self.calibration[horizon].observe(endpoint is not None,
                None if endpoint is None else float((endpoint-prediction).norm()))
            if endpoint is not None:
                if credit is None:
                    self.dynamics[horizon].observe(history, endpoint-origin)
                else:
                    self.dynamics[horizon].observe(history, endpoint-origin, credit=credit)
        forecasts = []
        self.forecast_cache = {key: value for key, value in self.forecast_cache.items()
                               if key[1] > tracker.frame}
        for entity in entities:
            identity = entity['id']
            history = self.histories.setdefault(identity, deque(maxlen=5))
            history.append(observed.get(identity))
            if len(history) < 5 or any(p is None for p in history):
                continue
            positions = torch.stack(list(history))
            differences = positions[1:]-positions[:-1]
            for horizon, model in self.dynamics.items():
                credit = None
                if hasattr(model, 'predict_with_credit'):
                    displacement, credit = model.predict_with_credit(differences)
                else:
                    displacement = (model.predict(differences) if len(model.keys)
                                    else horizon*differences[-1])
                prediction = positions[-1]+displacement
                forecast = dict(id=identity, horizon_samples=horizon,
                    origin_sample=tracker.frame, evidence_age=0,
                    position=prediction, **self.calibration[horizon].summary())
                forecasts.append(forecast)
                self.forecast_cache[identity, tracker.frame+horizon] = forecast
                if learn:
                    self.pending.setdefault(tracker.frame+horizon, []).append(
                        (identity, horizon, differences.clone(), positions[-1].clone(), prediction.clone(), credit))
        unobserved = {entity['id'] for entity in entities if not entity['observed']}
        for (identity, endpoint), forecast in self.forecast_cache.items():
            if identity in unobserved:
                forecasts.append({**forecast, 'horizon_samples': endpoint-tracker.frame,
                                  'evidence_age': tracker.frame-forecast['origin_sample']})
        self.compact()
        return dict(sample=tracker.frame, entities=entities, forecasts=forecasts,
                    support_field=field)

    def save(self, path):
        """Save learned parameters; loading intentionally starts a new scene."""
        memory = self.state.memory
        if type(memory) not in (PatchAssociation, LocalMetricAssociation) or any(
                type(m) not in (LocalMotionDynamics, ContextualMotionDynamics)
                for m in self.dynamics.values()):
            raise ValueError('unsupported candidate component type')
        torch.save(dict(version=2, state_type=type(self).__name__,
            memory_type=type(memory).__name__,
            dynamics_types={h: type(m).__name__ for h, m in self.dynamics.items()},
            height=self.observer.height, width=self.observer.width,
            observer=dict(weights=self.observer.weights, bias=self.observer.bias),
            memory=dict(keys=memory.keys, values=memory.values, metric=memory.metric),
            dynamics={h: dict(keys=m.keys, values=m.values, counts=m.counts)
                      for h, m in self.dynamics.items()},
            calibration={h: dict(outcomes=list(c.outcomes), residuals=list(c.residuals))
                         for h, c in self.calibration.items()}), path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, weights_only=True)
        if data['version'] != 2:
            raise ValueError('unsupported candidate checkpoint version')
        kinds = {kind.__name__: kind for kind in (StreamingVisualState,
            ContrastNormalizedVisualState, NoiseCalibratedVisualState,
            ConsensusVisualState, LearnedVisualCandidate)}
        name = data['state_type']
        if name not in kinds:
            raise ValueError('unsupported candidate observation architecture')
        memory_types = {kind.__name__: kind for kind in (PatchAssociation, LocalMetricAssociation)}
        dynamics_types = {kind.__name__: kind for kind in (LocalMotionDynamics, ContextualMotionDynamics)}
        if data['memory_type'] not in memory_types or any(
                kind not in dynamics_types for kind in data['dynamics_types'].values()):
            raise ValueError('unsupported candidate component type')
        memory = memory_types[data['memory_type']](dimensions=data['memory']['keys'].shape[1])
        dynamics = {h: dynamics_types[kind]() for h, kind in data['dynamics_types'].items()}
        model = kinds[name](height=data['height'], width=data['width'], memory=memory, dynamics=dynamics)
        for name, value in data['observer'].items():
            setattr(model.observer, name, value)
        for name, value in data['memory'].items():
            setattr(model.state.memory, name, value)
        for horizon, values in data['dynamics'].items():
            for name, value in values.items():
                setattr(model.dynamics[horizon], name, value)
        for horizon, values in data['calibration'].items():
            model.calibration[horizon].outcomes.extend(values['outcomes'])
            model.calibration[horizon].residuals.extend(values['residuals'])
        return model


class ContrastNormalizedVisualState(StreamingVisualState):
    """Explicit grayscale gain control; foreground semantics remain unlabelled."""

    def frame_contrast(self, visible):
        contrast = visible-visible.median()
        scale = torch.quantile(contrast.abs(), .999)
        if scale == 0:
            scale = contrast.abs().max()
        return (contrast/scale.clamp(min=torch.finfo(contrast.dtype).eps)).clamp(-1., 1.)


class NoiseCalibratedVisualState(ContrastNormalizedVisualState):
    """Gain control with a robust Gaussian sensor-noise floor."""

    def frame_contrast(self, visible):
        contrast = visible-visible.median()
        sigma = contrast.abs().median()/.67448975
        scale = torch.maximum(torch.quantile(contrast.abs(), .999), 6*sigma)
        if scale == 0:
            scale = contrast.abs().max()
        return (contrast/scale.clamp(min=torch.finfo(contrast.dtype).eps)).clamp(-1., 1.)


class ConsensusObservationTracker(FrameObservationTracker):
    def accept_match(self, history, position, sensory):
        if history['hits'] >= 3:
            return True
        if float((position-history['position']).norm()) <= .5:
            return True
        y, x = (round(float(v)) for v in position)
        evidence = sensory[12:14] if len(sensory) >= 14 else sensory[:2]
        return bool((evidence[:, max(0, y-2):y+3, max(0, x-2):x+3] > .2).any())


class ConsensusVisualState(NoiseCalibratedVisualState):
    """New moving image associations require nearby event corroboration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state.tracker = ConsensusObservationTracker(height=self.observer.height,
                                                         width=self.observer.width)
        self.reset_state()


class LearnedVisualCandidate(ConsensusVisualState):
    def __init__(self, *, height=32, width=64, observer=None, memory=None, dynamics=None):
        super().__init__(height=height, width=width, observer=observer,
            memory=memory if memory is not None else LocalMetricAssociation(),
            dynamics=dynamics if dynamics is not None else
            {h: ContextualMotionDynamics() for h in (1, 4, 8)})
