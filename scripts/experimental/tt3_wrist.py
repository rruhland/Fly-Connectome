"""A fourth rate actuator pitches the blade; control channels are independently shuffled."""

import math
import numpy as np

from tt3_arena import TT3Arena, _clamp


class WristArena(TT3Arena):
    def __init__(self, seed=0, *, view_seed=None, wiring_seed=0, **kwargs):
        rng = np.random.default_rng(wiring_seed)
        self.order = rng.permutation(4)
        self.signs = rng.choice((-1.0, 1.0), 4)
        super().__init__(seed, view_seed=view_seed, **kwargs)

    def reset(self, *, render=True):
        self.pitch = 0.0
        self.wrist_rate = 0.0
        return super().reset(render=render)

    def _advance_arm(self, delta, dt):
        super()._advance_arm(delta, dt)
        self.pitch = _clamp(self.pitch + dt * 2.6 * self.wrist_rate, (-0.48, 0.48))
        base, elbow, center, normal = self._pose
        nx, ny, nz = normal
        c, s = math.cos(self.pitch), math.sin(self.pitch)
        self._pose = base, elbow, center, (nx, ny * c - nz * s, ny * s + nz * c)

    def step(self, action, *, render=True):
        action = np.asarray(action, dtype=float)
        if (
            action.shape != (4,)
            or not np.isfinite(action).all()
            or max(abs(action)) > 1
        ):
            raise ValueError("supply four finite bounded rates")
        body = self.signs * action[self.order]
        self.wrist_rate = body[-1]
        return super().step(body[:3], render=render)
