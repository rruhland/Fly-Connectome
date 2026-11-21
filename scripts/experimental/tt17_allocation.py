"""Apply learned travel limits in calibrated actuator coordinates during a game."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, minimize

from .tt6_body_control import MOTOR_BUDGET
from .tt8_service_drive import ServiceDriveServo
from .tt14_gain_servo import GainServo


def limit_travel(command, matrix, native_position):
    """Project an already rate-bounded command onto the learned travel envelope."""
    travel = native_position + matrix @ command
    if np.max(np.abs(travel)) <= MOTOR_BUDGET + 1e-9:
        return command
    limits = LinearConstraint(matrix, -MOTOR_BUDGET-native_position, MOTOR_BUDGET-native_position)
    result = minimize(lambda u: .5*np.sum((u-command)**2), np.zeros(len(command)), jac=lambda u:u-command,
                      bounds=Bounds(-1.,1.), constraints=[limits], method='SLSQP',
                      options={'maxiter':20,'ftol':1e-10})
    if result.success and np.max(np.abs(native_position + matrix @ result.x)) <= MOTOR_BUDGET + 1e-7:
        return np.clip(result.x,-1.,1.)
    return np.zeros_like(command)


class NativeTravelServo(GainServo):
    """Use the original mixed response planner, changing only the travel coordinates."""
    def advance(self, preceding):
        if preceding is None:
            return
        self.commands[1:] = self.commands[:-1]
        self.commands[0] = preceding
        self.requested += preceding
        mixed = self.commands @ self.transform.T
        self.effective = self.transform @ self.requested + np.sum(mixed * (self.fit.kernels.T-1.), axis=0)
        self.observer.advance(self.fit.pose(self.effective)[:3])
        self.nominal_history = [self.effective.copy(), *self.nominal_history[:-1]]

    def _solve_plan(self, normal, rhs):
        wanted = np.linalg.solve(normal, rhs)[:self.fit.action_size]
        return limit_travel(np.clip(wanted,-1.,1.),self.transform,self.transform @ self.requested)


class AllocatedServo(ServiceDriveServo):
    """Plan in the familiar motor coordinates and allocate bounded public rates."""
    def __init__(self, motor, memory, *, learn):
        super().__init__(motor.base,memory,learn=learn)
        self.transform = motor.transform
        self.inverse = np.linalg.pinv(self.transform)

    def advance(self, preceding):
        mixed = None if preceding is None else self.transform @ preceding
        super().advance(mixed)

    def command(self, goal, lead, pitch):
        native_rate = super().command(goal, lead, pitch)
        raw = self.inverse @ native_rate
        if np.max(np.abs(raw)) > 1.:
            raw = lsq_linear(self.transform,native_rate,bounds=(-1.,1.),method='bvls').x
        return limit_travel(np.clip(raw,-1.,1.),self.transform,self.requested)
