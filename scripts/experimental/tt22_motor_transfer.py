"""Use a reward-selected clock stroke and a transferred visual position goal."""

import numpy as np

from .tt19_factory import FactoryModel, HOME, RATE
from .tt19_flight import optical_start
from .tt20_curriculum import CLAMP_SHIFT
from .tt22_openloop import stroke


def lateral_goal(ready, mover, velocity, coefficients):
    """Project Pong's relative displacement and motion terms onto table width."""
    _, displacement, motion = coefficients
    lookahead = max(0., min(90., (ready[1]-mover[1])/velocity[1]))
    return float(ready[0] + displacement*(mover[0]-ready[0]) + motion*lookahead*velocity[0]), lookahead


class MotorPrimitive:
    """Drive the existing timed primitive after kinematic translation of its keyframes."""

    def __init__(self, parameters, coefficients):
        self.parameters = np.asarray(parameters, dtype=float).copy()
        self.coefficients = coefficients
        self.target = np.zeros(5)
        self.history = []
        self.observed_ready = None
        self.decision = None
        self.factory = FactoryModel()
        self.factory.base += CLAMP_SHIFT
        self.ready_joints = HOME + RATE[:5]*self.parameters[:5]
        self.peak_joints = self.ready_joints + RATE[:5]*self.parameters[5:10]
        self.ready_world = self.factory.pose(self.ready_joints)[0]

    def watching(self, tick):
        return self.coefficients is not None and self.decision is None and tick < self.parameters[10]

    def _translate(self, goal, ready):
        dx = goal - ready[0]
        achieved, translated = [], []
        for joints in (self.ready_joints, self.peak_joints):
            target, normal, _, _ = self.factory.pose(joints)
            target[0] += dx
            moved = self.factory.solve(target, normal, joints)
            achieved.append(dict(requested=target.tolist(), solved=self.factory.pose(moved)[0].tolist()))
            translated.append(moved)
        self.parameters[:5] = (translated[0]-HOME)/RATE[:5]
        self.parameters[5:10] = (translated[1]-translated[0])/RATE[:5]
        return achieved

    def command(self, tick, ball=None, blade=None):
        if self.watching(tick):
            if self.observed_ready is None and blade is not None:
                self.observed_ready = blade.copy()
            self.history.append((tick, ball))
            estimate = optical_start(self.history, .0109)
            if estimate is not None and tick >= 4:
                mover, speed = estimate
                if .025 < mover[1] < .38 and speed[1] < -.026:
                    ready = self.ready_world if self.observed_ready is None else self.observed_ready
                    goal, horizon = lateral_goal(ready, mover, speed, self.coefficients)
                    self.decision = dict(tick=tick, goal_x=goal, displacement_m=float(goal-ready[0]),
                                         horizon_frames=float(horizon), mover=mover.tolist(), speed=speed.tolist(),
                                         ready=ready.tolist(), rgb_ready=self.observed_ready is not None,
                                         keyframes=self._translate(goal, ready))
        request = np.clip(stroke(self.parameters, tick)-self.target, -1., 1.)
        self.target += request
        return np.r_[request, 0.]
