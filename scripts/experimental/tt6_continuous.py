"""Commission changing bodies from public camera probes and play through hidden disturbances."""
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .tt5_vision import motion_tracker_factory
from .tt6_body_vision import body_tracker_factory
from .tt6_body_control import MOTOR_BUDGET, CameraBodyPlayer, CameraServo, _samples, fit_camera_body
from .tt6_body_probe import collect_probe, probe_commands
from .tt6_timing import VisualDelay, advance_stroke


def prediction_error(adapter, history, factory):
    """Test an old map on a complete new intervention, including blade inclination."""
    commands, indices, poses = _samples([history], factory)
    if len(poses) < 10:
        return None
    requested = np.cumsum(commands[0], axis=0)
    effective = requested + np.column_stack([lfilter(kernel - 1., [1.], commands[0, :, j])
                                              for j, kernel in enumerate(adapter.kernels)])
    predictions = adapter.pose(effective[indices[1]])
    actual = poses[:, :3]
    absolute = np.linalg.norm(predictions[:, :3] - actual, axis=1)
    movement = np.linalg.norm(actual - adapter.ready, axis=1)
    visible = np.isfinite(poses[:, 3])
    if adapter.pitch_controllable and visible.sum() >= 5:
        absolute[visible] += .10 * np.abs(predictions[visible,3] - poses[visible,3])
    return dict(error=float(np.median(absolute)), upper=float(np.quantile(absolute, .80)),
                stationary_error=float(np.median(movement)), sightings=len(poses))


def supports(adapter, evidence):
    return (evidence is not None and evidence['error'] < min(.085, max(.035, 1.65 * adapter.error)) and
            evidence['upper'] < .135 and evidence['error'] < .63 * evidence['stationary_error'])


@dataclass
class BodyLibrary:
    models: list

    def commission(self, wiring, rig, schedule, *, horizon=34, probe_cache=None):
        histories, probes, attempts = [], [], []
        size = wiring.action_size
        old = [(i, m) for i,m in enumerate(self.models) if m.action_size == size]
        collect = collect_probe
        if probe_cache is not None:
            from .tt7_probe_cache import ProbeCache

            collect = ProbeCache(probe_cache, wiring, rig, schedule, horizon).collect
        for i, seed in enumerate(schedule['calibration']):
            commands = probe_commands(schedule['player_seed'] + i, horizon, size)
            history = collect(wiring, seed, commands)
            histories.append(history)
            probes.append(dict(seed=seed, actions=[a.tolist() for a,_ in history[1:]]))
            n = len(probes)
            if n not in (1, 2, 3, 5, len(schedule['calibration'])):
                continue
            if n <= 2 and not old:
                continue
            for make_tracker in (body_tracker_factory, motion_tracker_factory):
                try:
                    factory = make_tracker(rig, histories)
                    if n <= 2 and old:
                        scored = []
                        for index, model in old:
                            evidence = prediction_error(model, history, factory)
                            if supports(model, evidence):
                                scored.append((evidence['error'], index, model))
                        if scored:
                            _, index, selected = min(scored, key=lambda entry: entry[0])
                            attempts.append(dict(probes=n, optics=make_tracker.__name__, recalled=index))
                            return selected, factory, probes, attempts
                    if n < 3:
                        continue
                    if n < len(schedule['calibration']):
                        trial = fit_camera_body(histories[:-1], factory)
                        if not supports(trial, prediction_error(trial, history, factory)):
                            continue
                    fitted = fit_camera_body(histories, factory)
                    attempts.append(dict(probes=n, optics=make_tracker.__name__, learned=len(self.models)))
                    self.models.append(fitted)
                    return fitted, factory, probes, attempts
                except ValueError as err:
                    attempts.append(dict(probes=n, optics=make_tracker.__name__, error=str(err)))
        return None, None, probes, attempts


class DelayedCameraServo(CameraServo):
    def __init__(self, fit, memory, *, learn=True):
        super().__init__(fit)
        self.observer = VisualDelay(memory, fit.ready, np.array([.015,.035,.018]), learn=learn)

    def advance(self, preceding):
        super().advance(preceding)
        if preceding is not None:
            self.observer.advance(self.fit.pose(self.effective)[:3])

    def observe(self, actor, pitch):
        self.observer.observe(actor.position if actor is not None and actor.observed else None)
        super().observe(actor, pitch)


class PredictiveServo(DelayedCameraServo):
    """Use the identified delay and learned step response to plan a short sequence of rates."""

    horizon = 8
    moves = 3

    def __init__(self, fit, memory, *, learn=True):
        super().__init__(fit, memory, learn=learn)
        self.nominal_history = [self.effective.copy() for _ in self.observer.belief]
        self.intercept = None
        self.stroke = .049
        self.weights = np.array([1., 1., 1., .13])
        self.scale = np.linspace(.65, 1., self.horizon)

    def advance(self, preceding):
        self.effective = self.nominal_history[0].copy()
        super().advance(preceding)
        if preceding is not None:
            self.nominal_history = [self.effective.copy(), *self.nominal_history[:-1]]

    def observe(self, actor, pitch):
        self.observer.observe(actor.position if actor is not None and actor.observed else None)
        self.effective = self.observer.belief @ np.asarray(self.nominal_history)
        CameraServo.observe(self, actor, pitch)

    def command(self, goal, lead, pitch):
        # Predict the next few camera poses if outstanding commands settle, then
        # solve the local linear response for three bounded future rate commands.
        fit = self.fit
        depth = fit.kernels.shape[1]
        future = []
        for step in range(self.horizon + 1):
            transient = fit.kernels[:, step:].T - 1.
            future.append(self.requested + np.sum(self.commands[:depth-step] * transient, axis=0))
        free = np.stack([*reversed(self.nominal_history[1:]), *future])
        delay = int(np.clip(round(self.observer.delay), 0, len(self.nominal_history)-1))
        times = np.arange(1, self.horizon + 1)
        states = free[len(self.nominal_history) - 1 + times - delay]
        age = times[:, None] - np.arange(self.moves) - delay - 1
        effect = np.where(age[..., None] >= 0, fit.kernels.T[np.maximum(age, 0)], 0.)
        jac = fit.jacobian(self.effective) * self.weights[:, None]
        design = np.einsum('t,an,tmn->tamn', self.scale, jac, effect)
        design = design.reshape(4*self.horizon, self.moves * fit.action_size)
        targets = np.tile(np.r_[goal, pitch], (self.horizon, 1))
        if self.intercept is not None:
            initial = np.clip(self.stroke * self.intercept, -.107, .145)
            targets[:, 1] += initial - np.clip(self.stroke*(self.intercept-times), -.107, .145)
        error = (targets - fit.pose(states) - self.correction) * self.weights * self.scale[:, None]
        normal = design.T @ design + .00055 * np.eye(design.shape[1])
        rhs = design.T @ error.ravel()
        self._strike_objective(normal, rhs, states, effect, jac, targets)
        plan = np.linalg.solve(normal, rhs)
        lower = np.maximum(-1., -MOTOR_BUDGET - self.requested)
        upper = np.minimum(1., MOTOR_BUDGET - self.requested)
        return np.clip(plan[:fit.action_size], lower, upper)


    def _strike_objective(self, normal, rhs, states, effect, jac, targets):
        """Optional additional requirements on the predicted movement at impact."""


class ContinualPlayer(CameraBodyPlayer):
    """Share shot-response and latency evidence across anonymous receiving arms."""

    def __init__(self, adapter, prior, factory, response, timing, *, adjust=True, predictive=False):
        self.response = response
        self.timing_memory = timing
        self.adjust = adjust
        self.predictive = predictive
        super().__init__(adapter, prior, factory, retain_trace=True)

    def _make_servo(self):
        if self.predictive:
            return PredictiveServo(self.adapter, self.timing_memory, learn=self.adjust)
        if self.adjust:
            return DelayedCameraServo(self.adapter, self.timing_memory)
        return super()._make_servo()

    def reset(self):
        super().reset()
        self.stroke_speed = self.response.speed

    def _stroke_goals(self, estimate):
        goal, lead, pitch, horizon = super()._stroke_goals(estimate)
        if self.predictive:
            self.servo.intercept = horizon
            self.servo.stroke = self.stroke_speed
        elif self.adjust and horizon is not None:
            goal, lead = advance_stroke(goal, lead, horizon, self.stroke_speed, self.servo.observer.delay)
        return goal, lead, pitch, horizon

    def finish(self, reward):
        details = super().finish(reward)
        details['impact'] = self.response.finish([frame['ball'] for frame in self.trace], reward)
        if self.adjust:
            observer = self.servo.observer
            observer.finish(carry=True)
            details['timing'] = dict(estimated=observer.delay, informative=observer.informative)
        return details
