import numpy as np
import pytest

from scripts.experimental.tt29_motor_correction import (
    ContactPlan,
    contact_shift,
    edited_goals,
)


def test_contact_shift_compares_flights_at_the_same_depth():
    path = np.array([[0.10, 0.25, -1.20, 1.05], [0.14, 0.27, -1.32, 1.07]])
    anchor = ContactPlan(2.10, np.array([0.22, -1.26, 1.04]), np.zeros(4))
    reach_time, delay, displacement = contact_shift(path, anchor, now=2.0)
    assert reach_time == pytest.approx(0.12)
    assert delay == pytest.approx(0.02)
    np.testing.assert_allclose(displacement, [0.04, 0.0, 0.02])
    assert contact_shift(path[:1], anchor, now=2.0) is None


def test_fractional_phase_and_local_translation_preserve_the_commanded_start():
    base = np.arange(1, 17)[:, None] * np.ones((1, 6))
    target = np.zeros(6)
    zeros = np.zeros(5)
    np.testing.assert_array_equal(
        edited_goals(base, target, 0.0, zeros, 0.08, 0.02), base
    )
    delayed = edited_goals(base, target, 0.01, zeros, 0.08, 0.02)
    advanced = edited_goals(base, target, -0.01, zeros, 0.08, 0.02)
    np.testing.assert_allclose(delayed[0], 0.5)
    np.testing.assert_allclose(advanced[0], 1.5)
    translated = edited_goals(base, target, 0.0, np.full(5, 0.1), 0.08, 0.02)
    np.testing.assert_allclose(translated[3, :5], base[3, :5] + 0.1)
    np.testing.assert_array_equal(translated[:, 5], base[:, 5])
    np.testing.assert_array_equal(translated[-1], base[-1])
