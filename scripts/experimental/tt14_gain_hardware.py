"""Evaluator-side changes to motor gain and cross-axis coupling."""

import numpy as np

from .tt6_body_arena import TT6Wiring


def perturbation(width, seed, profile):
    """Return the environment's transform from requested rates to original port rates."""
    if profile == 'stable':
        return np.eye(width)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 19309]))
    if profile == 'gain':
        gains = rng.uniform(.61, .94, width)
        gains[rng.integers(width)] = .51
        return np.diag(gains)
    if profile != 'coupling':
        raise ValueError(f'unknown motor perturbation {profile!r}')
    matrix = np.diag(rng.uniform(.65, .76, width))
    order = rng.permutation(width)
    signs = rng.choice((-1, 1), width)
    matrix[order, np.roll(order, 1)] = signs * (.98 - matrix[order, order])
    return matrix


class GainWiring(TT6Wiring):
    """Mix the port rates after transport noise and before the normal physical mapping."""

    def __init__(self, arena, seed, matrix=None):
        super().__init__(arena, seed)
        self.matrix = np.eye(arena.action_size) if matrix is None else np.asarray(matrix, dtype=float)
        if self.matrix.shape != (arena.action_size, arena.action_size) or not np.isfinite(self.matrix).all():
            raise ValueError('expected a finite square motor transform')

    def step(self, action, *, render=True):
        return super().step(self.matrix @ action, render=render)
