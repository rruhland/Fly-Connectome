"""Carry a robust optical arrival estimate across short disruptions of the observed moving ball."""

from collections import deque

import numpy as np


class ArrivalHistory:
    def __init__(self):
        self.estimates = deque()
        self.last_position = None
        self.last_time = -10.0

    def update(self, now, raw):
        if raw is None:
            return None
        raw = np.asarray(raw).copy()
        # A changed flight or prolonged gap starts a new track. Only past images enter the estimate.
        if now - self.last_time > 0.18 or (
            len(raw) > 6
            and self.last_position is not None
            and raw[6] > self.last_position + 0.35
        ):
            self.estimates.clear()
        self.last_position = raw[6] if len(raw) > 6 else None
        self.last_time = now
        while self.estimates and now - self.estimates[0][0] > 0.32:
            self.estimates.popleft()
        if raw[0] < 0.0 or raw[0] > 0.65:
            return raw
        reading = raw[:6].copy()
        reading[0] += now
        self.estimates.append((now, reading))
        observations = np.array([r for _, r in self.estimates])
        valid = (observations[:, 2] > 0.76) & (observations[:, 2] < 1.45)
        if valid.sum() < 3:
            return raw
        # Near the ball the latest free-flight samples resolve its location.
        # Earlier, use agreement among independent exposures from the same flight.
        if raw[0] < 0.12:
            return raw
        center = np.median(observations[valid], axis=0)
        spread = np.array([0.025, 0.035, 0.05, 0.30, 0.50, 0.50])
        delta = (observations - center) / spread
        magnitude = np.mean(np.minimum(delta**2, 50.0), axis=1)
        w = valid / (0.24 + magnitude)
        result = w @ observations / w.sum()
        result[0] -= now
        raw[:6] = result
        return raw
