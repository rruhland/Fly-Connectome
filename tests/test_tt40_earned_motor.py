import numpy as np

from scripts.experimental.tt37_motor_clinic import schedules
from scripts.experimental.tt38_motor_value import MotorValue
from scripts.experimental.tt40_local_critic import Conditional
from scripts.experimental.tt40_paired_practice import motions, policies


def test_zero_intervention_preserves_the_whole_issued_sequence():
    rng = np.random.default_rng(13)
    fit = rng.normal(size=(2, 5))
    action = rng.normal(size=10)
    dt, eta = 1 / 60, 0.144
    home = np.zeros(6)
    frames = round((eta + 1.15) / dt)
    original = schedules(*fit, eta, action[None, :], dt, home, frames)
    actual = motions(fit, action, eta, dt, home, policies()[:1])
    np.testing.assert_array_equal(original, actual)


def test_local_outcome_update_requires_the_issued_movement_and_earned_score():
    prior = {"x": np.zeros((2, 11)), "y": [[0, 0], [1, 1]]}
    predicted = np.arange(13, dtype=float) * 0.05
    stock = MotorValue(prior)
    for allow in (True, False):
        model = Conditional(prior, {"examples": []}, update=allow)
        model.context(np.zeros(10))
        np.testing.assert_array_equal(
            model.predict(predicted), stock.predict(predicted)
        )
        model.issued(predicted)
        model.issued(None)
        model.feedback([], True)
        assert model.online == []
        model.issued(predicted)
        model.feedback([1], False)
        assert model.online == []
        model.feedback([-1], False)
        assert len(model.online) == int(allow)
        if allow:
            assert model.online[0][-1] == [1, 1]
        model.issued(predicted)
        model.feedback([], True)
        assert len(model.online) == 2 * int(allow)
        if allow:
            assert model.online[-1][-1] == [0, 0]
