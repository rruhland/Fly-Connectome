"""Choose a motor servo using only a camera-fitted model of its response."""

import numpy as np

from .tt4_arena import PADDLE_RADIUS
from .tt6_continuous import ContinualPlayer
from .tt6_rotary_adapt import LagAwareServo


def servo_evidence(adapter):
    """Measure the weakest spatial direction under a unit public motor request."""
    home_response = adapter.jacobian(np.zeros(adapter.action_size))[:3]
    gains = np.linalg.svd(home_response, compute_uv=False)
    threshold = .8 * PADDLE_RADIUS
    return dict(weakest_meters=float(gains[-1]), threshold_meters=threshold,
                choice='short' if gains[-1] < threshold else 'predictive')


class SelectingPlayer(ContinualPlayer):
    """Use one shot planner for both fitted-motor solvers."""

    def __init__(self, adapter, prior, factory, response, timing, *, servo='automatic', adjust=True):
        self.servo_evidence = servo_evidence(adapter)
        self.servo_choice = self.servo_evidence['choice'] if servo == 'automatic' else servo
        if self.servo_choice not in ('short', 'predictive'):
            raise ValueError(f'unknown motor servo {self.servo_choice!r}')
        super().__init__(adapter, prior, factory, response, timing, adjust=adjust,
                         predictive=self.servo_choice == 'predictive')

    def _make_servo(self):
        if self.servo_choice == 'short':
            servo = LagAwareServo(self.adapter, self.timing_memory)
            servo.observer.learn = self.adjust
            return servo
        return super()._make_servo()
