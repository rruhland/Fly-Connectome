import numpy as np
from scripts.experimental.tt105_generate_return_effects import (
    H,
    WIDTH,
    alternatives,
    choose,
)


def test_zero_and_grip_preserving_perturbation():
    rng = np.random.default_rng(123)
    goal = rng.normal(size=(H, 6))
    goal[:, 5] = -0.10
    np.testing.assert_array_equal(alternatives(goal, 0.1, np.zeros(WIDTH)), goal)
    motion = alternatives(goal, 0.1, np.ones(WIDTH))
    np.testing.assert_array_equal(motion[:, 5], goal[:, 5])


def test_strong_next_return_gain_must_keep_current_return():
    good = np.ones((3, 3), bool)
    score = np.array([[0.8] * 3, [0.9] * 3, [0.92] * 3])
    radius = np.array([[0.05] * 3, [0.045] * 3, [0.076] * 3])
    assert choose(score, radius, good) == 1
    good[1, 2] = False
    assert choose(score, radius, good) == 0
    good[1, 2] = True
    score[1, 1] = 0.799
    assert choose(score, radius, good) == 0
