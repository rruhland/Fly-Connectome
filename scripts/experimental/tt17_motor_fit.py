"""Bound continuous camera-body corrections using a repeatable optical movement."""

from dataclasses import dataclass

import numpy as np

from .tt13_live_ports import effective
from .tt12_motor_recall import rewire
from .tt14_gain_model import MotorMap, Observation, collect, fit_transform
from .tt16_coded_probe import decode


def coordinates(source):
    strengths = np.linalg.norm(source.jacobian(np.zeros(source.action_size)), axis=1)
    return np.clip(np.median(strengths[:3]) / np.maximum(strengths, .01), .10, 1.5)


@dataclass
class PairedMotion:
    moves: np.ndarray
    ticks: np.ndarray
    target: np.ndarray
    reference: np.ndarray
    difference: np.ndarray
    scale: np.ndarray

    @property
    def training(self):
        return (self.ticks // 3) % 2 == 0

    def loss(self, motor, mask=slice(None)):
        mapped = self.moves @ motor.transform.T
        prediction = motor.base.pose(effective(mapped, motor.base.kernels)[self.ticks])
        raw = (prediction - self.target)[mask] * self.scale
        return float(np.mean(np.minimum(.22, np.linalg.norm(np.nan_to_num(raw), axis=1))))


def align_observations(source, actions, reference, active):
    zero = np.zeros(source.action_size)
    moves = np.vstack((zero, actions))
    ticks, left, right = np.intersect1d(reference['ticks'], active['ticks'], return_indices=True)
    before = np.asarray(reference['seen'], dtype=float)[left]
    after = np.asarray(active['seen'], dtype=float)[right]
    difference = after - before
    initial = difference[ticks <= 1]
    if len(initial):
        # A constant difference between starting detections should not represent a change in motors.
        for axis in range(4):
            values = initial[:, axis][np.isfinite(initial[:, axis])]
            if len(values):
                difference[:, axis] -= np.median(values)
    baseline = source.pose(effective(moves, source.kernels)[ticks])
    return PairedMotion(moves, ticks, baseline + difference, baseline, difference, coordinates(source))


def constrained(matrix):
    diagonal = np.diag(matrix).copy()
    matrix = np.clip(matrix, -.42, .42)
    np.fill_diagonal(matrix, np.clip(diagonal, .42, 1.35))
    for _ in range(5):
        singular = np.linalg.svd(matrix, compute_uv=False)
        if .35 <= singular[-1] and singular[0] <= 1.5:
            return matrix
        # A clipped ridge solution can still be ill-conditioned. Shrink back toward the recalled basis.
        matrix = np.eye(len(matrix)) + .7 * (matrix - np.eye(len(matrix)))
    return np.eye(len(matrix))


def _fit(base, signal, kind, subset, initial):
    width = base.action_size
    mask = np.eye(width, dtype=bool)
    selected = mask if kind == 'gain' else np.ones_like(mask)
    penalty = np.where(mask, .025, .06)[selected]
    obs = Observation(signal.moves, signal.ticks[subset], signal.target[subset])
    responses = obs.responses(base)[0]
    observed = np.isfinite(obs.poses)
    hessian = np.zeros((width, width, 4))
    left, right = np.triu_indices(width)
    hessian[left, right] = base.coefficients[1 + width:]
    hessian += hessian.transpose(1, 0, 2).copy()
    identity = np.eye(width)
    transform = initial.copy()
    for _ in range(4):
        state = np.einsum('tij,ij->ti', responses, transform)
        residual = np.nan_to_num(obs.poses - base.pose(state)) * signal.scale
        robust = 1. / np.sqrt(1. + np.sum(residual**2, axis=1) / .055**2)
        jac = base.coefficients[1:width+1] + np.einsum('ti,ijo->tjo', state, hessian)
        design = np.einsum('tio,tij,o->toij', jac, responses, signal.scale)[..., selected]
        design *= (observed * robust[:, None])[..., None]
        design = design.reshape(-1, len(penalty))
        step = np.linalg.solve(design.T @ design + np.diag(penalty),
                               design.T @ (residual * robust[:, None]).ravel()
                               - penalty * (transform - identity)[selected])
        transform[selected] += np.clip(step, -.24, .24)
        transform = constrained(transform)
    return MotorMap(base, transform, kind)


def initialise(source, actions, reference, active):
    identity = np.eye(source.action_size)
    key, signed_evidence = decode(source, actions, active['ticks'], active['seen'])
    signed_key = [list(range(source.action_size)), [1]*source.action_size]
    base = source if key == signed_key else rewire(source, *key)
    signed = MotorMap(base, identity, 'original' if base is source else 'remap')
    alternate = MotorMap(source, identity, 'original')
    signal = align_observations(source, actions, reference, active)
    if len(signal.ticks) < 6:
        return signed, signed, signal, dict(signed=key, signed_evidence=signed_evidence, reason='few paired sightings')
    discrepancy = float(np.mean(np.linalg.norm(np.nan_to_num(signal.difference * signal.scale), axis=1)))
    event = dict(signed=key, signed_evidence=signed_evidence, matched_frames=len(signal.ticks),
                 apparent_change_m=discrepancy, signed_loss_m=signal.loss(signed))
    train = signal.training
    if discrepancy < .012:
        return signed, alternate, signal, dict(event, reason='unchanged paired response', selected_loss_m=discrepancy)
    if min(train.sum(), (~train).sum()) < 6:
        return signed, signed, signal, dict(event, reason='few independent optical blocks')
    baseline = min((signed, alternate), key=lambda model: signal.loss(model, train))
    candidates = [(signal.loss(baseline, train), baseline)]
    for model in ([alternate, signed] if signed.base is not source else [alternate]):
        for kind in ('gain', 'coupling'):
            fit = _fit(model.base, signal, kind, train, identity)
            loss = signal.loss(fit, train)
            if loss + .004 < signal.loss(baseline, train) and signal.loss(fit, ~train) + .004 < signal.loss(baseline, ~train):
                candidates.append((loss + (.003 if kind == 'coupling' else 0.), fit))
    _, selected = min(candidates, key=lambda item: item[0])
    if selected.coupled:
        selected = _fit(selected.base, signal, selected.kind, slice(None), selected.transform)
    return signed, selected, signal, dict(event, reason=selected.kind, selected_loss_m=signal.loss(selected))


class Refinement:
    """Adopt a local correction only if a later point and the dry movement both support it."""

    def __init__(self, motor, signal, enabled):
        self.motor = motor
        self.initial = motor
        self.signal = signal
        self.enabled = enabled and motor.coupled
        self.previous = None
        self.updates = 0

    def observe(self, actions, trace):
        event = dict(adopted=False)
        if not self.enabled:
            return event
        obs = collect(actions, trace, self.motor.base.action_size)
        if min(obs.train.sum(), (~obs.train).sum()) < 5:
            return event
        previous, self.previous = self.previous, obs
        if previous is None:
            return event
        current = self.motor.errors(obs)
        earlier = self.motor.errors(previous)
        if max(current) < .065 or min(earlier) < .055:
            return event
        trial = fit_transform(self.motor.base, [previous], self.motor.transform, self.motor.kind)
        if trial is None or np.linalg.norm(trial.transform - self.initial.transform) > .30:
            return event
        trial.transform = constrained(trial.transform)
        proposed = trial.errors(obs)
        supported = all(after + .012 < .87*before for after, before in zip(proposed, current, strict=True))
        anchored = self.signal.loss(trial) < self.signal.loss(self.motor) + .004
        if supported and anchored:
            self.motor = trial
            self.updates += 1
            self.previous = None
            event = dict(adopted=True, before=current, after=proposed)
        return event
