"""Frozen feedback policies for the randomized three-dimensional source worlds."""

from pathlib import Path

import numpy as np


CHECKPOINT = Path(__file__).resolve().parents[2] / 'artifacts' / 'tt24' / 'world-prior.npz'
SCHEMA = 'tt24-world-prior-1'
FEATURES = 10
COEFFICIENTS = 4
VECTORS = 4
WEIGHTS = FEATURES + 2 * COEFFICIENTS * VECTORS
LIMITS = np.array((.069, .039, .181))


def _bounded(vector, scale):
    return vector / max(scale, np.linalg.norm(vector))


def _vectors(state):
    return np.asarray([_bounded(state[:2] - state[6:8], .074), _bounded(state[3:5], .42),
                       _bounded(state[6:8], .071), _bounded(state[9:11], .19)])


class WorldPrior:
    """Map an 11-channel metric observation to five absolute task-space targets.

    Inputs are ball position and velocity, then paddle position and its upward
    normal's x/y components. Positions are in metres relative to blade home.
    Outputs are a requested position relative to that home and normal x/y.
    """

    def __init__(self, weights):
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (WEIGHTS,) or not np.isfinite(weights).all():
            raise ValueError(f'expected {WEIGHTS} finite learned coefficients')
        self.weights = weights.copy()
        self.vertical = self.weights[:FEATURES]
        self.horizontal = self.weights[FEATURES:].reshape(2, VECTORS, COEFFICIENTS)

    @classmethod
    def load(cls, path=CHECKPOINT):
        with np.load(path, allow_pickle=False) as archive:
            if archive['schema'].item() != SCHEMA:
                raise ValueError('unsupported source-world policy checkpoint')
            return cls(archive['weights'])

    def target(self, state):
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (11,) or not np.isfinite(state).all():
            raise ValueError('expected an 11-vector of finite metric observations')
        v = _vectors(state)
        z, vz, bz = np.clip((state[2] / .19, state[5] / 2.1, state[8] / .041), -1., 1.)
        r = float(v[0] @ v[0])
        s = float(v[1] @ v[1])
        features = np.array((1., z, vz, bz, z * vz, z * z, vz * vz, r, s, v[0] @ v[1]))
        common = features[[0, 1, 2, 7]]
        horizontal = (self.horizontal @ common) @ v
        horizontal *= LIMITS[[0, 2], None] / (1. + np.linalg.norm(horizontal, axis=1, keepdims=True))
        vertical = self.vertical @ features
        return np.r_[horizontal[0], LIMITS[1] * vertical / (1. + abs(vertical)), horizontal[1]]
