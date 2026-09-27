"""Production hybrid visual state; camera-sample horizons and explicit online credit."""
import math
from collections import deque
import torch
from .observation import LocalObservationModel, local_motion_map
from .association import StatisticalObservationTracker
from .memory import FrameObservationState, LocalMetricAssociation
from .dynamics import SpatialBelief, mixture_quantile

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
                         {h: SpatialBelief(horizon=h) for h in (1, 4, 8)})
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
        shape = (self.observer.height, self.observer.width)
        for value, expected in ((events, (2, *shape)), (visible_frame, shape)):
            if value is None and expected == shape:
                continue
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
                raise ValueError(f'sensor shape must be {expected}')
            if value.device.type != 'cpu':
                raise ValueError('the approved vision runtime requires CPU tensors')
            if value.is_complex() or not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
                raise ValueError('sensor values must be finite and in [0, 1]')
        events = events.float()
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


class ProbabilisticVisualState(StreamingVisualState):
    def __init__(self, **kwargs):
        kwargs.setdefault('dynamics', {h: SpatialBelief(horizon=h) for h in (1, 4, 8)})
        kwargs.setdefault('memory', LocalMetricAssociation())
        super().__init__(**kwargs)
        self.state.tracker = StatisticalObservationTracker(height=self.observer.height, width=self.observer.width)
        self.reset_state()


    def frame_contrast(self, visible):
        contrast = visible-visible.median()
        sigma = contrast.abs().median()/.67448975
        bound = math.sqrt(2*math.log(2*visible.numel()/.01))
        scale = torch.maximum(torch.quantile(contrast.abs(), .999), 2*bound*sigma)
        if scale == 0:
            scale = contrast.abs().max()
        return (contrast/scale.clamp(min=torch.finfo(contrast.dtype).eps)).clamp(-1., 1.)

    @torch.no_grad()
    def step(self, events, visible_frame, *, learn=False):
        result = super().step(events, visible_frame, learn=learn)
        for forecast in result['forecasts']:
            if forecast['evidence_age']:
                continue
            history = torch.stack(list(self.histories[forecast['id']]))
            model = self.dynamics[forecast['horizon_samples']]
            centers, weights = model.distribution(history[1:]-history[:-1])
            low, high = model.calibration.bounds()
            forecast.update(mixture_centers=centers+history[-1], mixture_weights=weights,
                component_sigma_pixels=1., point_semantics='mixture_mean',
                marginal_interval_90=torch.stack((mixture_quantile(centers, weights, low),
                                                  mixture_quantile(centers, weights, high)))+history[-1],
                interval_calibration_samples=len(model.calibration.ranks))
        return result

    def save(self, path):
        if type(self.state.memory) is not LocalMetricAssociation or any(
                type(m) is not SpatialBelief for m in self.dynamics.values()):
            raise ValueError('unsupported probabilistic candidate component')
        torch.save(dict(version=1, architecture='ProbabilisticVisualState',
            sensor='grayscale-every-sample-plus-events', height=self.observer.height,
            width=self.observer.width, observer=dict(weights=self.observer.weights, bias=self.observer.bias),
            memory={name: getattr(self.state.memory, name) for name in ('keys', 'values', 'metric')},
            dynamics={h: dict(capacity=m.capacity, keys=m.keys, values=m.values, seen=m.seen,
                generator=m.generator.get_state(), ranks=list(m.calibration.ranks)) for h, m in self.dynamics.items()},
            calibration={h: dict(outcomes=list(c.outcomes), residuals=list(c.residuals))
                         for h, c in self.calibration.items()}), path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, weights_only=True)
        if data['version'] != 1 or data['architecture'] != 'ProbabilisticVisualState':
            raise ValueError('unsupported probabilistic candidate checkpoint')
        dynamics = {h: SpatialBelief(capacity=d['capacity'], horizon=h) for h, d in data['dynamics'].items()}
        model = cls(height=data['height'], width=data['width'], dynamics=dynamics,
                    memory=LocalMetricAssociation(dimensions=data['memory']['keys'].shape[1]))
        for name, value in data['observer'].items():
            setattr(model.observer, name, value)
        for name, value in data['memory'].items():
            setattr(model.state.memory, name, value)
        for h, values in data['dynamics'].items():
            for name in ('keys', 'values', 'seen'):
                setattr(dynamics[h], name, values[name])
            dynamics[h].generator.set_state(values['generator'])
            dynamics[h].calibration.ranks.extend(values['ranks'])
        for h, values in data['calibration'].items():
            model.calibration[h].outcomes.extend(values['outcomes'])
            model.calibration[h].residuals.extend(values['residuals'])
        return model
