"""Predict the blade's motion through a learned mixing of motor rates."""

import numpy as np

from .tt6_body_control import MOTOR_BUDGET
from .tt8_service_drive import ServiceDriveServo


class GainServo(ServiceDriveServo):
    def __init__(self, motor, memory, *, learn):
        super().__init__(motor.base, memory, learn=learn)
        self.transform = motor.transform

    def advance(self, preceding):
        if preceding is None:
            return
        requested = np.clip(self.requested + preceding, -MOTOR_BUDGET, MOTOR_BUDGET)
        self.commands[1:] = self.commands[:-1]
        self.commands[0] = requested - self.requested
        self.requested = requested
        mixed = self.commands @ self.transform.T
        self.effective = self.transform @ requested + np.sum(mixed * (self.fit.kernels.T - 1.), axis=0)
        self.observer.advance(self.fit.pose(self.effective)[:3])
        self.nominal_history = [self.effective.copy(), *self.nominal_history[:-1]]

    def command(self, goal, lead, pitch):
        fit = self.fit
        depth = fit.kernels.shape[1]
        mixed = self.commands @ self.transform.T
        position = self.transform @ self.requested
        future = [position + np.sum(mixed[:depth-step] * (fit.kernels[:, step:].T - 1.), axis=0)
                  for step in range(self.horizon + 1)]
        free = np.stack([*reversed(self.nominal_history[1:]), *future])
        delay = int(np.clip(round(self.observer.delay), 0, len(self.nominal_history) - 1))
        times = np.arange(1, self.horizon + 1)
        states = free[len(self.nominal_history) - 1 + times - delay]
        age = times[:, None] - np.arange(self.moves) - delay - 1
        settled = np.where(age[..., None] >= 0, fit.kernels.T[np.maximum(age, 0)], 0.)
        effect = settled[..., None] * self.transform
        jac = fit.jacobian(self.effective) * self.weights[:, None]
        design = np.einsum('t,an,tmni->tami', self.scale, jac, effect)
        design = design.reshape(4 * self.horizon, self.moves * fit.action_size)
        targets = np.tile(np.r_[goal, pitch], (self.horizon, 1))
        if self.intercept is not None:
            initial = np.clip(self.stroke * self.intercept, -.107, .145)
            targets[:, 1] += initial - np.clip(self.stroke * (self.intercept - times), -.107, .145)
        error = (targets - fit.pose(states) - self.correction) * self.weights * self.scale[:, None]
        normal = design.T @ design + .00055 * np.eye(design.shape[1])
        rhs = design.T @ error.ravel()
        self._strike_objective(normal, rhs, states, effect, jac, targets)
        return self._solve_plan(normal, rhs)

    def _strike_objective(self, normal, rhs, states, effect, jac, targets):
        if self.drive is None or self.intercept is None or not 0. < self.intercept < self.horizon + 1.:
            return
        motion = np.diff(self.fit.pose(states)[:, 1])
        response = np.einsum('n,tmni->tmi', jac[1], np.diff(effect, axis=0))
        times = np.arange(2, self.horizon + 1) - .5
        weight = 2.8 * np.exp(-.5 * ((times - self.intercept) / 1.25)**2)
        rows = (response * weight[:, None, None]).reshape(len(times), -1)
        normal += rows.T @ rows
        rhs += rows.T @ ((self.drive - motion) * weight)
