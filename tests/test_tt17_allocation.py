import numpy as np

from scripts.experimental.tt17_allocation import limit_travel
from scripts.experimental.tt6_body_control import MOTOR_BUDGET


def test_learned_limits_apply_after_receiving_motor_rates_are_mixed():
    matrix = np.array([[.50,.30],[.0,.75]])
    position = np.array([MOTOR_BUDGET-.1,-MOTOR_BUDGET+.05])
    candidate = limit_travel(np.array([1.,-.8]), matrix, position)
    assert np.max(abs(position + matrix @ candidate)) <= MOTOR_BUDGET + 1e-7
    assert np.max(abs(candidate)) <= 1.
    assert np.linalg.norm(candidate - [1.,-.8]) < np.linalg.norm([1.,-.8])
    at_half_gain = np.eye(2)*.5
    np.testing.assert_equal(limit_travel(np.ones(2), at_half_gain, np.array([2.,0.])), [1.,1.])
