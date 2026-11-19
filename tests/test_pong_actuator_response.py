"""Numerical and carry checks for joint visual actuator response inference."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from pong_actuator_response import ActuatorResponseController, RESPONSE_TIMES, response_dynamics
from pong_rapid_motor_recovery import COAST_SECONDS, RESPONSE_SECONDS, motor_dynamics
from pong_timing_controller import TimedMotorController
from pong_transfer_arena import interception_policy
from pong_visual_button_effects import VisualButtonEffects


def test_response_projection_matches_continuous_transition():
    for response in (*RESPONSE_TIMES, COAST_SECONDS):
        for elapsed in (.016666666666666666, .025, .35):
            matrix = np.array([[0., 1., 0., 0.],
                               [0., -1. / COAST_SECONDS, 1. / COAST_SECONDS, 0.],
                               [0., 0., -1. / response, 1. / response],
                               [0., 0., 0., 0.]])
            expected = expm(elapsed * matrix)
            transition, drive = response_dynamics(elapsed, response)
            np.testing.assert_allclose(transition, expected[:3, :3], atol=1e-14)
            np.testing.assert_allclose(drive, expected[:3, 3], atol=1e-14)
            if response == RESPONSE_SECONDS:
                for actual, baseline in zip((transition, drive), motor_dynamics(elapsed), strict=True):
                    np.testing.assert_array_equal(actual, baseline)


def test_frozen_carried_response_keeps_continuous_value_through_delay_updates():
    policy = interception_policy()
    learner = VisualButtonEffects()
    state = TimedMotorController(policy, learner).state()
    state.update(response_hypothesis=.28, response_seconds=.329)
    encoded = json.loads(json.dumps(state))
    controller = ActuatorResponseController(policy, learner, timing_state=encoded, response_updates=False)
    assert controller.state() == state
    evidence = np.full(len(controller.hypotheses), -2.)
    evidence[controller.lags.index(12)] = 0.
    for _ in range(24):
        controller._adapt_response(evidence, .02)
    assert controller.delay_commands == 12
    assert controller.state()['response_hypothesis'] == .28
    assert controller.state()['response_seconds'] == .329
