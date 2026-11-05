"""Focused, non-native checks of generic three-request response learning."""
from collections import deque
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from pong_categorical_motor_fir import CategoricalMotorFIR, CategoricalMotorController
from pong_visual_motor_fir import DEFAULTS
from test_pong_visual_motor_fir import QuantizedMotor, feedback, inherited


def test_arbitrary_three_request_response_and_absent_history_are_distinct():
    learner = CategoricalMotorFIR(71, settings=dict(DEFAULTS, lags=4, horizon=3,
        initial_covariance=1e7, forgetting_seconds=10000.))
    # The formerly neutral request has its own one-interval delayed response.
    learner.theta[4 + 1] = .4
    assert learner.predict(learner.future_features([], 0)) == pytest.approx(.016)
    assert learner.predict(learner.future_features([0], 0)) == pytest.approx(.024)
    assert learner.predict(learner.future_features([], -1)) == 0.
    assert np.count_nonzero(learner.interval_features([], .02)) == 0
    assert learner.interval_features([0], .02)[4] == .02
    learner.theta[:] = 0.
    # An arbitrary nonmonotonic map, never supplied to the learner as a label.
    response = np.array([-.12, .37, -.27])
    history = deque(maxlen=4)
    rng = np.random.default_rng(718)
    for i in range(1100):
        history.append(int(rng.integers(3)) - 1)
        x = np.zeros((3, 4))
        dt = 1 / 60 if i % 2 else 1 / 40
        for lag, request in enumerate(reversed(history)):
            x[request + 1, lag] = dt
        target = dt * response[history[-3] + 1] if len(history) >= 3 else 0.
        learner.observe(x.ravel(), target, dt)
    for _ in range(40):
        history = rng.integers(-1, 2, 4).tolist()
        x = learner.interval_features(history, .02)
        assert learner.predict(x) == pytest.approx(.02 * response[history[-3] + 1], abs=2e-7)
    assert learner.ready([.02, .02, -.02])


def test_json_continuation_preserves_held_probe_rng_and_frozen_coefficients():
    live = CategoricalMotorFIR(16)
    rng = np.random.default_rng(91)
    for _ in range(300):
        x = rng.normal(size=len(live.theta)) * .02
        live.observe(x, .1 * x[40] - .3 * x[70], .02)
    for _ in range(2000):
        live.perturb(1, True)
        if live.probe_remaining:
            break
    assert live.probe_remaining > 0
    restored = CategoricalMotorFIR.from_state(json.loads(json.dumps(live.state())))
    for _ in range(80):
        assert live.perturb(1, True) == restored.perturb(1, True)
        x = rng.normal(size=len(live.theta)) * .02
        target = float(rng.normal() * .01)
        assert live.observe(x, target, .02) == restored.observe(x, target, .02)
    assert live.state() == restored.state()
    frozen = live.frozen_branch()
    assert isinstance(frozen, CategoricalMotorFIR)
    theta, covariance = frozen.theta.copy(), frozen.covariance.copy()
    for _ in range(20):
        frozen.perturb(1, True)
        frozen.observe(rng.normal(size=len(frozen.theta)) * .02, .1, .02)
    np.testing.assert_array_equal(frozen.theta, theta)
    np.testing.assert_array_equal(frozen.covariance, covariance)
    frozen.perturb(1, False)
    assert frozen.probe_remaining == 0


def test_public_quantized_cycle_is_relearned_without_a_mapping_input():
    learner = CategoricalMotorFIR(173)
    controller = CategoricalMotorController(inherited(), learner)
    plant = QuantizedMotor()
    # The chosen cycle challenge adds no command delay. A separate development
    # check with this helper's extra six-frame delay failed the readiness gate;
    # that harder combined challenge is recorded in the research note.
    plant.queue.clear()
    for i in range(1400):
        current = plant.observation(i)
        action, _ = controller.choose(current, 1)
        plant.step(action - 1)
        controller.observe(**feedback(), public_entities=plant.observation(i + 1))
    frozen = learner.frozen_branch()
    controls = [CategoricalMotorController(inherited(), learner),
                CategoricalMotorController(inherited(), frozen)]
    plants = [QuantizedMotor(), QuantizedMotor()]
    for plant in plants:
        plant.queue.clear()
    errors = [[], []]
    for i in range(3800):
        for j, (control, motor) in enumerate(zip(controls, plants)):
            current = motor.observation(i)
            requested, _ = control.choose(current, 1)
            # This mapping belongs only to the synthetic environment. Every
            # actuator response remains available; request1 now produces motion.
            delivered = (0., 1., -1.)[requested]
            motor.step(delivered)
            after = motor.observation(i + 1)
            control.observe(**feedback(), public_entities=after)
            if i >= 2300:
                errors[j].append((after[0]['normalized_position'][0]
                    - after[1]['normalized_position'][0]) ** 2)
    live_rmse, frozen_rmse = [float(np.sqrt(np.mean(e))) for e in errors]
    assert live_rmse < .06
    assert live_rmse < .4 * frozen_rmse
    assert controls[0].numeric_actions > 2500
    assert not np.array_equal(learner.theta, frozen.theta)
