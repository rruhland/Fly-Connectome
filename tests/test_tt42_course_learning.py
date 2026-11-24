import numpy as np

from scripts.experimental.tt42_optical_uncertainty import ForecastErrors
from scripts.experimental.tt42_paired_advantage import Advantages
from scripts.experimental.tt42_learning_course import PLAN
from scripts.experimental.tt4_transfer_trial import _read


def test_uncertainty_only_learns_after_subsequent_observations():
    history = ForecastErrors()
    start = np.array([0.05, -0.4, 1.08])
    rate = np.array([0.2, -3.1, 0.6])
    accel = np.array([0.0, 0.0, -9.81]) - 0.13 * np.linalg.norm(rate) * rate
    for i in range(5):
        t = i / 30
        point = start + t * rate + 0.5 * t**2 * accel
        estimate = np.r_[point, rate + t * accel + [0.0, 0.0, 0.1]]
        history.observe(t, point, estimate)
        if i < 4:
            assert not history.errors
    assert len(history.errors) == 1 and history.errors[0][5] < -0.08
    history.observe(t, point, estimate)
    assert len(history.errors) == 1
    history.reset()
    assert history.last is None and len(history.errors) == 1


def test_paired_learning_cannot_credit_the_action_for_an_easy_feed():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 12, 7))
    y = np.repeat([[0], [1], [1], [0], [1]], 12, axis=1)
    learner = Advantages.__new__(Advantages)
    learner.fit(x, y)
    np.testing.assert_allclose(learner.weights, 0)


def test_prospective_course_keeps_all_draws_and_releases_disjoint():
    d = _read(PLAN)
    cards = d["practice"] + d["assessment"]
    assert len(d["practice"]) == 48 and len(d["assessment"]) == 24
    assert len({c["id"] for c in cards}) == 72
    assert all(c["opening"] == "random" and c["profile"] == "reach" for c in cards)
    assert len({c["reply_seed"] for c in cards}) == 72
