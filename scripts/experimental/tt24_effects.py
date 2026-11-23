"""Identify local actuator effects from actions and successive encoder readings."""

from collections import deque

import numpy as np


class MotorEffects:
    """Predict encoder displacement for a held command and solve its local inverse.

    An observation is (command, joint displacement, previous joint displacement).
    No action-to-joint correspondence is assumed. All arrays use the public
    interface order. A short window lets new experience replace obsolete mappings.
    """

    def __init__(self, channels, *, blocks=18, ridge=.003):
        self.channels = channels
        self.rows = deque(maxlen=blocks)
        self.targets = deque(maxlen=blocks)
        self.ridge = ridge

    def update(self, command, displacement, previous):
        self.rows.append(np.r_[command, previous, 1.])
        self.targets.append(np.asarray(displacement).copy())

    def coefficients(self):
        x = np.asarray(self.rows)
        y = np.asarray(self.targets)
        penalty = self.ridge * np.ones(x.shape[1])
        penalty[self.channels:-1] *= 12.
        return np.linalg.solve(x.T @ x + np.diag(penalty), x.T @ y)

    def predict(self, command, previous):
        return np.r_[command, previous, 1.] @ self.coefficients()

    def solve(self, desired, previous):
        coef = self.coefficients()
        effect = coef[:self.channels]
        drift = np.r_[previous, 1.] @ coef[self.channels:]
        precision = effect @ effect.T + self.ridge * np.eye(self.channels)
        return np.linalg.solve(precision, effect @ (desired - drift))


def transmission(rng):
    q, r = np.linalg.qr(rng.normal(size=(5, 5)))
    return q @ np.diag(np.sign(np.diag(r)) * rng.uniform(.75, 1.08, 5))


def excitation(rng, passes=1):
    # Each direction is followed by its negative to limit drift with unknown wiring.
    probes = []
    for _ in range(passes):
        directions, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        for direction in directions:
            probes.extend((.44 * direction, -.44 * direction))
    return probes
