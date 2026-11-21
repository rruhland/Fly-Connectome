"""A camera-rate pursuit servo using the table player's learned forward model."""
import numpy as np

from .tt6_body_control import CameraServo, MOTOR_BUDGET
from .tt10_target import read_markers
from .tt11_tracking import orientation


class PursuitServo:
    def __init__(self, fit, *, forecast=True):
        self.fit = fit
        self.forecast = forecast
        self.servo = CameraServo(fit)
        self.horizon = 6
        self.pulse_response = fit.kernels[:, :self.horizon].sum(axis=1)
        self.view_evidence = .0004 * np.eye(3)
        self.rotation = np.eye(3)
        self.sample = None
        self.goal = None
        self.goal_velocity = np.zeros(3)
        self.action = None
        self.tick = 0

    def act(self, rgb):
        pos, goal = read_markers(rgb)
        self.servo.advance(self.action)
        current = self.fit.pose(self.servo.effective)[:3]
        if self.tick % 4 == 0:
            if self.sample is not None:
                old_pos, old_prediction = self.sample
                self.view_evidence += np.outer(pos - old_pos, current - old_prediction)
                self.rotation = orientation(self.view_evidence)
            self.sample = (pos, current)
        if self.goal is not None:
            self.goal_velocity = .65 * self.goal_velocity + .35 * (goal - self.goal)
        self.goal = goal
        target = goal + self.horizon * self.goal_velocity if self.forecast else goal
        future = self.servo.requested + np.sum(
            self.servo.commands[:-self.horizon] * (self.fit.kernels[:, self.horizon:].T - 1.), axis=0)
        drift = self.fit.pose(future)[:3] - current
        required = self.rotation.T @ (target - pos) - drift
        jac = self.fit.jacobian(future)[:3] * self.pulse_response
        cmd = jac.T @ np.linalg.solve(jac @ jac.T + .037**2 * np.eye(3), required)
        # At most two command units per channel in any eight consecutive ticks,
        # the same bound as the block controller. Physical excursions stay equal.
        self.action = np.clip(cmd, np.maximum(-.25, -MOTOR_BUDGET-self.servo.requested),
                              np.minimum(.25, MOTOR_BUDGET-self.servo.requested))
        self.tick += 1
        return self.action
