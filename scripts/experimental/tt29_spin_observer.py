"""Estimate the change in ball spin from a visible tabletop bounce."""

import numpy as np

from .tt4_arena import BALL_RADIUS, BALL_MASS, BALL_INERTIA, SPIN_DECAY
from .tt29_vision_twin import OpticalPlanner


class SpinPlanner(OpticalPlanner):
    def __init__(self, twin, tracker, *, seed):
        super().__init__(twin, seed=seed)
        self.tracker = tracker
        self.spin_updates = []

    def _refresh(self, step, flight, angles, rate, target):
        super()._refresh(step, flight, angles, rate, target)
        observed = self.tracker
        if (
            not flight.bounced
            or observed.bounce_state is None
            or observed.bounce_time is None
        ):
            return
        age = step * self.twin.control_dt - observed.bounce_time
        incoming, outgoing = observed.bounce_state
        if not 0 <= age < 0.35 or incoming[1] >= -1.0 or outgoing[1] >= -0.8:
            return
        delta = outgoing - incoming
        spin = (BALL_MASS * BALL_RADIUS / BALL_INERTIA) * np.array(
            [delta[1], -delta[0], 0.0]
        )
        spin = np.clip(spin * np.exp(-SPIN_DECAY * age), -85.0, 85.0)
        i = self.twin._ball_dof + 3
        self.twin.data.qvel[i : i + 3] = spin
        self.spin_updates.append([step, *spin.tolist()])
