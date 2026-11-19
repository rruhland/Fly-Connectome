"""Verify the timing wrapper preserves the inherited control and carries learned state."""

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from pong_rapid_motor_recovery import RESPONSE_SPEEDS, RapidMotorController, motor_dynamics
from pong_timing_controller import TimedMotorController
from test_pong_visual_button_effects import calibrated
from test_relational_persistent_binding import entity, model


def test_frozen_zero_lag_keeps_the_stage_one_request_and_posterior():
    learner = calibrated((.11, .04, -.13))
    policy = model()
    rapid = RapidMotorController(policy, learner)
    timed = TimedMotorController(policy, learner, timing_updates=False)

    def scene(paddle, target):
        return [entity(10, paddle, -.3), entity(11, target, 0.), entity(12, 0., .3)]

    for i in range(100):
        before = scene(.2 * math.sin(i / 10), .2 * math.cos(i / 10))
        after = scene(.2 * math.sin((i + 1) / 10), .2 * math.cos((i + 1) / 10))
        action, decision = rapid.choose(before, 1)
        request, diagnostic = timed.choose(before, 1)
        assert request == action
        assert {key: diagnostic[key] for key in decision} == decision
        update = dict(own_hits=0, reward=0., terminated=False, truncated=False,
                      elapsed_seconds=.02, public_entities=after)
        rapid.observe(**update)
        timed.observe(**update)
        np.testing.assert_array_equal(timed.belief.states, rapid.belief.states)
        np.testing.assert_array_equal(timed.belief.covariances, rapid.belief.covariances)
        np.testing.assert_array_equal(timed.belief.probabilities, rapid.belief.probabilities)
        assert timed.effective_requests == rapid.effective_requests
        assert timed.delay_commands == 0


def test_observed_delay_and_motor_belief_can_be_carried_then_restored():
    learner = calibrated((.11, .04, -.13))
    controller = TimedMotorController(model(), learner)
    for belief in controller.shadows.values():
        belief.initialize(0.)
    transition, drive = motor_dynamics(.02)
    history = []
    motion = np.zeros(3)
    for i in range(500):
        hidden_lag = 0 if i < 100 else 12
        history.append((i // 23 + i // 112) % 3)
        motion = transition @ motion + drive * RESPONSE_SPEEDS[history[-hidden_lag - 1]]
        if abs(motion[0]) > .3997:
            motion[0] = math.copysign(.3997, motion[0])
            motion[1] = 0.
        observations = np.array([
            belief.observe(history[-lag - 1] if i >= lag else 1, .02, motion[0])
            for lag, belief in controller.shadows.items()])
        controller._adapt_delay(observations, .02)
    assert controller.delay_commands == 12
    checkpoint = json.loads(json.dumps(controller.state()))
    frozen = TimedMotorController(model(), learner, timing_state=checkpoint, timing_updates=False)
    assert frozen.state() == checkpoint
    assert frozen.belief.states is None
    assert frozen.lags == (12,)
    assert frozen.effective_requests == controller.effective_requests


@pytest.mark.parametrize('live', (True, False))
def test_movement_before_carried_queue_matures_reconsiders_delay(live):
    learner = calibrated((.11, .04, -.13))
    state = TimedMotorController(model(), learner).state()
    state['delay_commands'] = 12
    controller = TimedMotorController(model(), learner, timing_state=state, timing_updates=live)

    def scene(position):
        return [entity(10, position, -.3), entity(11, -.25, 0.), entity(12, 0., .3)]

    action, _ = controller.choose(scene(0.), 1)
    assert action == 2
    controller.observe(own_hits=0, reward=0., terminated=False, truncated=False,
                       elapsed_seconds=.02, public_entities=scene(-.02))
    assert controller.delay_commands == (0 if live else 12)
