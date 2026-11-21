from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('mujoco')
from scripts.experimental.tt6_body_control import CameraBody, RESPONSE_FRAMES
from scripts.experimental.tt6_continuous import PredictiveServo
from scripts.experimental.tt6_timing import TimingMemory
from scripts.experimental.tt10_window import (
    BladeForecast, FACE_DISTANCE, Strike, WindowPlayer, _crossing_forecast,
)


def test_forecast_preserves_the_command_and_respects_delayed_motion():
    coefficients = np.zeros((15, 4))
    coefficients[0] = [0., -1.50, .3, .3]
    coefficients[1:5] = np.diag([.07, .062, .066, .17])
    kernels = np.tile(1. - .75 ** np.arange(1, RESPONSE_FRAMES + 1), (4, 1))
    body = CameraBody(coefficients, np.ones(4), kernels, .02, 30, 30, True)
    servo = PredictiveServo(body, TimingMemory([0., 0., 1., 0., 0., 0.]), learn=False)
    servo.weights[3] = .2
    strike = Strike(5.4, np.array([.06, -1.37, .35]), np.zeros(3), .52)
    servo.intercept = strike.time
    rate = servo.stroke
    first = servo.command(strike.target(strike.time, rate), strike.target(strike.time - 1, rate), strike.pitch)
    predicted = BladeForecast(servo).poses(strike, rate)
    delta = first * kernels[:, 0]
    np.testing.assert_allclose(predicted[:3], np.tile(body.home, (3, 1)))
    np.testing.assert_allclose(predicted[3], body.pose(delta))
    assert predicted[5, 1] > predicted[3, 1]


def test_face_crossing_reports_interpolated_time_and_blade_velocity():
    strike = Strike(1., np.array([.02, -1.38, .3]), np.zeros(3), 0.)
    ball = np.array([[.02, -1.24, .3, 0., -.1, 0.],
                     [.02, -1.34, .3, 0., -.1, 0.],
                     [.02, -1.44, .3, 0., -.1, 0.]])
    blade = np.array([[0., -1.50, .29, 0.],
                      [0., -1.45, .29, 0.],
                      [0., -1.40, .29, 0.]])
    result = _crossing_forecast(ball, blade, strike, .8, (0., 0.))
    assert result.time == pytest.approx(1 + (.11 - FACE_DISTANCE) / .15)
    assert result.blade_velocity == pytest.approx((0., .05, 0.))
    assert result.miss == pytest.approx(np.hypot(.02, .01))
    assert np.isfinite(result.cost)


def test_reply_gate_requires_visible_far_side_reversal_after_recognized_toss():
    player = WindowPlayer.__new__(WindowPlayer)
    player.scene = SimpleNamespace(tick=0, tracker=SimpleNamespace(ball=None))
    player.serve_plan = None
    player.far_depth = player.reply_tick = None

    def observe(tick, depth, seen=True):
        player.scene.tick = tick
        player.scene.tracker.ball = SimpleNamespace(observed=seen, position=(0., depth, .3))
        player._observe_reply()

    observe(33, 1.)
    assert player.far_depth is None
    player.serve_plan = (15., None)
    observe(18, .7)
    assert player.far_depth is None
    observe(35, 1.1)
    observe(36, .99)
    observe(37, .90, seen=False)
    assert player.reply_tick is None
    observe(38, .90)
    assert player.reply_tick == 38
