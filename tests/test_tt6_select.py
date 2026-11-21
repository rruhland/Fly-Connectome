from types import SimpleNamespace

import numpy as np

import pytest

pytest.importorskip('mujoco')

from scripts.experimental.tt4_transfer import TransferPrior
from scripts.experimental.tt6_body_control import CameraBody
from scripts.experimental.tt6_continuous import PredictiveServo
from scripts.experimental.tt6_racket import RacketAdaptation
from scripts.experimental.tt6_rotary_adapt import LagAwareServo
from scripts.experimental.tt6_select import SelectingPlayer, servo_evidence
from scripts.experimental.tt6_timing import TimingMemory


def _map(response):
    return SimpleNamespace(action_size=response.shape[1], jacobian=lambda _: response)


def test_servo_uses_spatial_control_response_not_wiring_or_channel_count():
    response = np.array([[.10, 0., 0., 0.], [0., .07, 0., 0.], [0., 0., .095, 0.], [0., 0., 0., .22]])
    strong = servo_evidence(_map(response))
    assert strong['choice'] == 'predictive'
    response[1, 1] = .05
    weak = servo_evidence(_map(response))
    assert weak['choice'] == 'short'
    extended = np.column_stack((response * [-1., 1., -1., 1.], np.zeros(4)))[:, [4, 2, 3, 0, 1]]
    np.testing.assert_allclose(servo_evidence(_map(extended))['weakest_meters'], weak['weakest_meters'])
    assert servo_evidence(_map(extended))['choice'] == weak['choice']


def test_auto_player_selects_the_actual_servo_at_reset():
    def player(weakest, adjust=True):
        coefficients = np.zeros((15, 4))
        coefficients[0] = [0., -1.4, .16, .27]
        coefficients[1:5] = np.diag([.085, weakest, .10, .17])
        adapter = CameraBody(coefficients, np.full(4, .5), np.ones((4, 14)), .01, 200, 200, True)
        return SelectingPlayer(adapter, TransferPrior('fresh', {}, 0), lambda: None,
                               RacketAdaptation(), TimingMemory(), adjust=adjust)

    assert isinstance(player(.07).servo, PredictiveServo)
    short = player(.05)
    assert isinstance(short.servo, LagAwareServo)
    short.reset()
    assert isinstance(short.servo, LagAwareServo)
    assert not player(.05, adjust=False).servo.observer.learn
