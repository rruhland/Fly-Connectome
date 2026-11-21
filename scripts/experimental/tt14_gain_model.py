"""Identify small motor-map corrections from requested rates and RGB blade sightings."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .tt6_body_control import POSE_WEIGHT
from .tt13_live_ports import error, evaluate, local_model, proposals, rewire


LAGS = np.arange(5)


@dataclass
class Observation:
    moves: np.ndarray
    ticks: np.ndarray
    poses: np.ndarray

    @property
    def train(self):
        return (self.ticks // 4) % 2 == 0

    def responses(self, model):
        integrated = self.moves.cumsum(axis=0)
        # Each physical output direction settles with its own recalled kernel.
        response = np.stack([integrated + lfilter(k - 1., [1.], self.moves, axis=0)
                             for k in model.kernels], axis=1)
        return response[np.maximum(self.ticks[None, :] - LAGS[:, None], 0)]

    def state(self):
        return dict(moves=self.moves.tolist(), ticks=self.ticks.tolist(),
                    poses=[[float(v) if np.isfinite(v) else None for v in row] for row in self.poses])

    @classmethod
    def restore(cls, state):
        return cls(np.array(state['moves']), np.array(state['ticks']), np.array(state['poses'], dtype=float))


@dataclass
class MotorMap:
    base: object
    transform: np.ndarray
    kind: str

    @property
    def coupled(self):
        return self.kind in ('gain', 'coupling')

    def errors(self, obs):
        if not self.coupled:
            return evaluate(self.base, obs.moves, obs.ticks, obs.poses, obs.train)
        predictions = self.base.pose(np.einsum('ltij,ij->lti', obs.responses(self.base), self.transform))
        train_costs = error(predictions[:, obs.train], obs.poses[obs.train])
        lag = np.argmin(train_costs)
        return float(train_costs[lag]), float(error(predictions[lag, ~obs.train], obs.poses[~obs.train]))


def collect(actions, trace, width):
    if len(trace) != len(actions) + 1:
        raise ValueError('unaligned visual and motor history')
    moves = np.vstack([np.zeros(width), actions])
    poses = np.array([[*row['blade'], row['blade_pitch']] if row['blade'] is not None else [None] * 4
                      for row in trace], dtype=float)
    ticks = np.flatnonzero(np.isfinite(poses[:, :3]).all(axis=1))
    return Observation(moves, ticks, poses[ticks])


def fit_transform(base, observations, seed, kind):
    """Solve regularized local least squares over the recalled nonlinear geometry."""
    width = base.action_size
    identity = np.eye(width)
    diagonal = kind == 'gain'
    selected = np.eye(width, dtype=bool) if diagonal else np.ones((width, width), dtype=bool)
    penalty = np.where(np.eye(width, dtype=bool), .025, .075)[selected]
    samples = [obs.responses(base)[:, obs.train] for obs in observations]
    poses = np.concatenate([obs.poses[obs.train] for obs in observations])
    known = np.isfinite(poses)
    # The Jacobian of the quadratic forward model is affine in its state.
    quadratic = np.zeros((width, width, 4))
    left, right = np.triu_indices(width)
    quadratic[left, right] = base.coefficients[1 + width:]
    quadratic += quadratic.transpose(1, 0, 2).copy()
    best = None
    for lag in range(len(LAGS)):
        response = np.concatenate([s[lag] for s in samples], axis=0)
        transform = seed.copy() if not diagonal else np.diag(np.diag(seed))
        for _ in range(3):
            states = np.einsum('tij,ij->ti', response, transform)
            residual = np.nan_to_num(poses - base.pose(states)) * POSE_WEIGHT
            weight = 1. / np.sqrt(1. + np.sum(residual**2, axis=1) / .055**2)
            jac = base.coefficients[1:1 + width] + np.einsum('ti,ijo->tjo', states, quadratic)
            design = np.einsum('tio,tij,o->toij', jac, response, POSE_WEIGHT)[..., selected]
            design *= (known * weight[:, None])[..., None]
            design = design.reshape(-1, len(penalty))
            step = np.linalg.solve(design.T @ design + np.diag(penalty),
                                   design.T @ (residual * weight[:, None]).ravel()
                                   - penalty * (transform - identity)[selected])
            transform[selected] += np.clip(step, -.28, .28)
        singular = np.linalg.svd(transform, compute_uv=False)
        if singular[-1] < .36 or singular[0] > 1.55:
            continue
        fit = error(base.pose(np.einsum('tij,ij->ti', response, transform)), poses)
        if best is None or fit < best[0]:
            best = float(fit), transform
    if best is not None:
        return MotorMap(base, best[1], kind)
    return None


class GainMemory:
    def __init__(self, source, mode, recalled=()):
        self.source = source
        self.mode = mode
        self.identity = np.eye(source.action_size)
        self.maps = [MotorMap(source, self.identity, 'original')]
        if mode in ('remap', 'hybrid'):
            self.maps.extend(MotorMap(model, self.identity, 'remap') for model in recalled)
        self.active = 0
        self.previous = None
        self.support = []

    @staticmethod
    def improves(current, candidate, *, strong=False):
        ratio, margin = (.63, .029) if strong else (.86, .012)
        return all(after < ratio * before and after + margin < before
                   for before, after in zip(current, candidate, strict=True))

    def observe(self, actions, trace):
        obs = collect(actions, trace, self.source.action_size)
        record = dict(previous=self.active, active=self.active, bank=len(self.maps))
        if min(obs.train.sum(), (~obs.train).sum()) < 5:
            return dict(record, reason='insufficient sightings')
        prior, self.previous = self.previous, obs
        if self.mode == 'fixed':
            return dict(record, reason='fixed')
        active = self.maps[self.active]
        current = active.errors(obs)
        record['errors'] = current
        if max(current) < .065:
            if active.coupled:
                self.support = [*self.support[-2:], obs]
            return dict(record, reason='within tolerance')
        choices = list(self.maps)
        if self.mode == 'remap':
            for key in proposals(self.source, obs.moves, obs.ticks[obs.train], obs.poses[obs.train]):
                choices.append(MotorMap(rewire(self.source, *key), self.identity, 'remap'))
        elif self.mode == 'refit':
            fresh = local_model(self.source, obs.moves, obs.ticks[obs.train], obs.poses[obs.train])
            choices.append(MotorMap(fresh, self.identity, 'refit'))
        else:
            # Recall is cheap and can recognize the original map on restoration.
            # Fit around the calibrated map only when recall has not explained this point.
            if not self.improves(current, self.maps[0].errors(obs)):
                for kind in ('gain', 'coupling'):
                    correction = fit_transform(self.source, [obs], active.transform, kind)
                    if correction is not None:
                        choices.append(correction)

        # Choose on alternating training blocks before checking the held blocks.
        # Prefer the smaller gain-only fit when the extra coefficients explain little.
        scores = [candidate.errors(obs) for candidate in choices]
        penalties = {'gain': .003, 'coupling': .008}
        best = min(range(len(choices)), key=lambda i: scores[i][0] + (
            penalties.get(choices[i].kind, 0.) if i >= len(self.maps) else 0.))
        proposed = choices[best]
        record.update(candidate=proposed.kind, candidate_errors=scores[best])
        if not self.improves(current, scores[best]):
            return dict(record, reason='insufficient improvement')

        if best >= len(self.maps) and proposed.coupled:
            supported = prior is not None and self.improves(active.errors(prior), proposed.errors(prior))
            if not supported and not self.improves(current, scores[best], strong=True):
                return dict(record, reason='waiting for another informative point')
            if supported:
                corroborating = [*self.support[-2:], prior] if active.coupled else [prior]
                pooled = fit_transform(self.source, [*corroborating, obs], proposed.transform, proposed.kind)
                if pooled is not None and self.improves(current, pooled.errors(obs)):
                    proposed = pooled
            self.support = [prior, obs] if supported else [obs]
        elif best != self.active:
            self.support = []

        if best >= len(self.maps):
            self.maps.append(proposed)
            best = len(self.maps) - 1
        self.active = best
        return dict(record, active=self.active, bank=len(self.maps), reason='switched',
                    selected_errors=proposed.errors(obs))
