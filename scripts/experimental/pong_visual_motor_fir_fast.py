"""Batch the existing FIR forecasts without changing their summation order.

This optional implementation is not installed into an already pinned run.
Each candidate retains the same time-ordered floating-point additions and
individual dot product as VisualMotorFIR.candidate_displacements.
"""
from functools import lru_cache

import numpy as np

from pong_visual_motor_fir import VisualMotorFIR


@lru_cache(maxsize=16)
def _indices(lags, horizon):
    indices = lags + np.arange(horizon)[:, None] - np.arange(lags)[None, :]
    indices.flags.writeable = False
    return indices


class FastVisualMotorFIR(VisualMotorFIR):
    def candidate_displacements(self, commands):
        lags, horizon = self.settings['lags'], self.settings['horizon']
        past = np.asarray(([0.] * lags + list(commands))[-lags:], dtype=float)
        history = np.empty((3, lags + horizon), dtype=float)
        history[:, :lags] = past
        history[:, lags:] = np.asarray([-1., 0., 1.])[:, None]
        terms = np.ones((3, horizon, lags + 1), dtype=float)
        terms[:, :, 1:] = history[:, _indices(lags, horizon)]
        terms *= self.settings['nominal_seconds']
        totals = np.cumsum(terms, axis=1)[:, -1, :]
        return [self.predict(row) for row in totals]
