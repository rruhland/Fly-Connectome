"""Opt-in open-loop Pong M1A run for directly observed local context efficacy."""
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import time

import torch

from full_context_efficacy import (AlwaysOpenContextPrediction, MultiContextEfficacyNetwork,
                                   MultiContextTimedPrediction)
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.plasticity import LearningConfig
from fly_connectome.pong import Physics, Pong
from fly_connectome.sensor import EventCamera, Retina


NETWORK_STATE = ('history', 'voltage', 'sensory_state', 'feedforward_current',
    'predictive_current', 'behavioral_current', 'adaptation', 'refractory',
    'magnitudes', 'excitatory_prediction', 'inhibitory_prediction', 'components',
    'context_exc', 'context_inh', 'current_context')
RULE_STATE = ('keys', 'values', 'arrival_trace', 'post_trace', 'expected', 'rates',
    'proposals', 'homeostatic_exponent', 'reference', 'previous_state', 'gate',
    'quiet_count', 'event_count', 'context_proposals')
PONG_STATE = ('rng', 'ball', 'ball_velocity', 'opponent', 'opponent_velocity', 'rally_steps')
BODY_STATE = ('position', 'activation', 'velocity')


def _tensors(obj, names):
    return {name: getattr(obj, name).clone() for name in names}


def _restore(obj, values):
    for name, value in values.items():
        setattr(obj, name, value.clone())


def adjacent_quiet_alarms(forecasts, events, targets):
    """Quiet target cells neighboring an event in the same cell, excluding edges."""
    samples = alarms = 0
    for frame in range(1, len(events)-1):
        current = torch.zeros(targets, dtype=torch.bool)
        current[events[frame][0]] = True
        neighboring = torch.zeros_like(current)
        neighboring[events[frame-1][0]] = True
        neighboring[events[frame+1][0]] = True
        eligible = neighboring & ~current
        samples += int(eligible.sum())
        alarms += int((forecasts[frame-1][eligible].abs() >= .1).sum())
    return dict(samples=samples, false_alarms=alarms,
                fraction=alarms/samples if samples else None)


class OpenLoopContextRun:
    def __init__(self, payload, seed, eta, *, components=None, warmup=True,
                 always_open=False):
        m = payload['metadata']
        if m['config']['stage'] != 'M1A' or m['config']['neural_steps'] != 8:
            raise ValueError('context bridge requires eight-tick open-loop M1A')
        self.seed, self.eta, self.frame = int(seed), float(eta), 0
        self.always_open = bool(always_open)
        self.retina = Retina(**m['retina'])
        graph = Graph(**m['graph'], gain=m['gain'])
        self.learning = replace(LearningConfig(**m['learning']), eta_prediction=eta,
                                eta_reward=0., homeostasis_rate=0.)
        self.sensory_gain = m['config']['sensory_gain']
        source_weights = payload['state']['network']['magnitudes']
        self.initial_weights = source_weights.clone().clamp_(0, self.learning.maximum_weight)
        self.net = MultiContextEfficacyNetwork(graph, m['delays'], m['pathways'],
            config=NeuronConfig(**m['neurons']), cell_types=m['retina']['cell_types'],
            target_mask=self.retina.injected)
        self.net.set_weights(self.initial_weights, components)
        if warmup:
            for _ in range(m['config']['warmup_steps']):
                self.net.step(torch.zeros_like(self.net.voltage))
        rule_type = AlwaysOpenContextPrediction if self.always_open else MultiContextTimedPrediction
        self.rule = rule_type(self.net, self.learning,
            sensory_mask=self.retina.injected, sensory_gain=self.sensory_gain)
        self.camera = EventCamera(1, self.retina.spec['height'], self.retina.spec['width'])
        self.pong = Pong([seed], Physics(**m['physics']))
        self.previous_prediction = None
        self.previous_target = None
        self.metrics = {name: dict(samples=0, squared_error=0., persistence_error=0.,
                                   anticipation=0., false_alarms=0)
                        for name in ('all', 'on', 'off', 'quiet')}
        self.trace = []
        self.events = []
        self.forecasts = []
        self.spike_counts = torch.zeros(self.net.n, dtype=torch.long)
        self.peak_spikes_per_tick = 0
        self.max_keys = 0
        self.open_gate_issues = 0
        self.max_update_error = 0.
        self.elapsed_seconds = 0.

    def _score(self, target):
        prediction, persistence = self.previous_prediction, self.previous_target
        if prediction is None:
            return
        error = (target-prediction).square()
        masks = dict(all=torch.ones_like(target, dtype=torch.bool), on=target < 0,
                     off=target > 0, quiet=target == 0)
        row = dict(frame=self.frame, on=int(masks['on'].sum()), off=int(masks['off'].sum()),
                   quiet=int(masks['quiet'].sum()),
                   squared_error=float(error.sum()),
                   event_squared_error=float(error[~masks['quiet']].sum()),
                   quiet_false_alarms=int((prediction[masks['quiet']].abs() >= .1).sum()))
        self.trace.append(row)
        for name, mask in masks.items():
            count = int(mask.sum())
            if count:
                stats = self.metrics[name]
                stats['samples'] += count
                stats['squared_error'] += float(error[mask].sum())
                stats['persistence_error'] += float((target[mask]-persistence[mask]).square().sum())
                stats['anticipation'] += float((prediction[mask]*target[mask]).sum())
                if name == 'quiet':
                    stats['false_alarms'] += row['quiet_false_alarms']

    @torch.no_grad()
    def step(self):
        started = time.perf_counter()
        image = self.pong.render(self.retina.spec['height'], self.retina.spec['width'])
        injection = self.retina.project(self.camera.observe(image))*self.sensory_gain
        issue = None
        for tick in range(8):
            activity = self.net.step(injection if tick == 0 else torch.zeros_like(injection),
                                     capture_increments=True)
            if tick == 0:
                target = self.learning.observation(activity, self.net.config.threshold,
                    self.retina.injected, self.sensory_gain)[0, self.net.targets]
                self._score(target)
                event_positions = (target != 0).nonzero().flatten()
                self.events.append((event_positions.tolist(), target[event_positions].tolist()))
            self.rule.observe(activity, torch.zeros(1))
            self.max_keys = max(self.max_keys, len(self.rule.keys))
            self.spike_counts.add_(activity.spikes[0].long())
            self.peak_spikes_per_tick = max(self.peak_spikes_per_tick, int(activity.spikes.sum()))
            if tick == 0:
                issue = self.rule.last_issue['prediction'].clone()
                self.open_gate_issues += int(self.rule.last_issue['gate'].sum())
        proposed = (self.net.components+self.rule.context_proposals).clamp(0, self.learning.maximum_weight)
        self.rule.synchronize()
        self.max_update_error = max(self.max_update_error,
                                    float((proposed-self.net.components).abs().max()))
        self.previous_prediction, self.previous_target = issue, target.clone()
        self.forecasts.append(issue)
        self.pong.step(-(self.pong.ball[:, 1]-self.pong.body.position)*10)
        self.frame += 1
        self.elapsed_seconds += time.perf_counter()-started
        if not torch.isfinite(self.net.voltage).all() or not torch.isfinite(self.net.components).all():
            raise ValueError('nonfinite neural state or context magnitude')

    def run(self, frames):
        for _ in range(frames):
            self.step()
        return self.summary()

    def summary(self):
        outside = torch.ones(self.net.e, dtype=torch.bool)
        outside[self.net.incoming] = False
        torch.testing.assert_close(self.net.magnitudes[outside], self.initial_weights[outside],
                                   rtol=0, atol=0)
        scores = {}
        for name, stats in self.metrics.items():
            count = stats['samples']
            scores[name] = dict(samples=count,
                mse=stats['squared_error']/count if count else None,
                persistence_mse=stats['persistence_error']/count if count else None,
                anticipation=stats['anticipation']/count if count else None,
                false_alarm_fraction=stats['false_alarms']/count if name == 'quiet' and count else None)
        duration = self.frame*8*self.net.config.dt
        return dict(seed=self.seed, eta=self.eta, frames=self.frame,
            always_open=self.always_open, seconds=self.elapsed_seconds,
            frames_per_second=self.frame/self.elapsed_seconds if self.elapsed_seconds else None,
            scores=scores,
            event_adjacent_quiet=adjacent_quiet_alarms(self.forecasts, self.events,
                                                        len(self.net.targets)),
            max_active_eligibility_keys=self.max_keys,
            open_gate_issues=self.open_gate_issues,
            peak_spikes_per_tick=self.peak_spikes_per_tick,
            mean_cell_rate_hz=float(self.spike_counts.float().mean()/duration) if duration else None,
            max_cell_rate_hz=float(self.spike_counts.max()/duration) if duration else None,
            changed_context_magnitudes=int((self.net.components !=
                self.initial_weights[self.net.incoming].repeat(2, 1)).sum()),
            max_update_reconstruction_error=self.max_update_error,
            non_target_magnitudes_unchanged=True)

    def save(self, path, source_sha):
        state = dict(schema_version=1, source_sha256=source_sha, seed=self.seed, eta=self.eta,
            always_open=self.always_open,
            frame=self.frame, network=_tensors(self.net, NETWORK_STATE),
            network_step=self.net.step_index, rule=_tensors(self.rule, RULE_STATE),
            rule_tick=self.rule.tick, forecast=self.rule.forecast,
            last_issue=self.rule.last_issue, last_confirmation=self.rule.last_confirmation,
            pong=_tensors(self.pong, PONG_STATE), body=_tensors(self.pong.body, BODY_STATE),
            camera_previous=self.camera.previous.clone(),
            previous_prediction=self.previous_prediction, previous_target=self.previous_target,
            metrics=self.metrics, trace=self.trace, events=self.events,
            forecasts=self.forecasts,
            spike_counts=self.spike_counts.clone(), peak_spikes_per_tick=self.peak_spikes_per_tick,
            max_keys=self.max_keys, open_gate_issues=self.open_gate_issues,
            max_update_error=self.max_update_error, elapsed_seconds=self.elapsed_seconds)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.tmp')
        os.close(fd)
        try:
            torch.save(state, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path, payload, source_sha):
        state = torch.load(path, weights_only=True)
        if state['schema_version'] != 1 or state['source_sha256'] != source_sha:
            raise ValueError('context run checkpoint/source mismatch')
        result = cls(payload, state['seed'], state['eta'], warmup=False,
                     always_open=state.get('always_open', False))
        _restore(result.net, state['network'])
        result.net.step_index = state['network_step']
        _restore(result.rule, state['rule'])
        result.rule.tick = state['rule_tick']
        result.rule.forecast = state['forecast']
        result.rule.last_issue = state['last_issue']
        result.rule.last_confirmation = state['last_confirmation']
        _restore(result.pong, state['pong'])
        _restore(result.pong.body, state['body'])
        result.camera.previous.copy_(state['camera_previous'])
        result.previous_prediction = state['previous_prediction']
        result.previous_target = state['previous_target']
        result.frame, result.metrics = state['frame'], state['metrics']
        result.trace, result.events = state['trace'], state['events']
        result.forecasts = state.get('forecasts', [])
        result.spike_counts = state['spike_counts']
        result.peak_spikes_per_tick = state['peak_spikes_per_tick']
        result.max_keys = state['max_keys']
        result.open_gate_issues = state['open_gate_issues']
        result.max_update_error = state['max_update_error']
        result.elapsed_seconds = state['elapsed_seconds']
        return result
