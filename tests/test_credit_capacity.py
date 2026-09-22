import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from credit_capacity import design, fit, score


def test_context_features_are_nested_without_new_bias_or_sign_flip():
    x = np.array([[1., -2.], [3., -4.], [5., -6.]])
    state = np.array([-1., 0., 1.])
    split = design(x, state, True)
    weights = np.array([.2, .3])
    np.testing.assert_allclose(split@np.tile(weights, 2), x@weights)
    np.testing.assert_array_equal(split[0, 2:], 0)
    np.testing.assert_array_equal(split[1, 2:], 0)
    np.testing.assert_array_equal(split[2, :2], 0)


def test_bounded_fit_recovers_context_specific_positive_magnitudes():
    x = design(np.ones((4, 1)), np.array([-1, -1, 1, 1]), True)
    weights, info = fit(x, np.array([.2, .2, .8, .8]), 1., False)
    np.testing.assert_allclose(weights, [.2, .8], atol=1e-8)
    assert info['success']
    bounded, _ = fit(np.ones((2, 1)), np.full(2, 2.), .5, False)
    np.testing.assert_allclose(bounded, [.5])


def test_zero_forecast_cannot_pass_by_getting_quiet_frames_right():
    y = np.r_[-1., 1., np.zeros(100)]
    m = score(y, np.zeros_like(y))
    assert m['false_alarm_fraction'] == 0
    assert m['on_anticipation'] == m['off_anticipation'] == 0


def test_prediction_is_clipped_for_reported_scores():
    m = score(np.array([-1., 1., 0.]), np.array([-2., 2., .2]))
    assert m['on_anticipation'] == m['off_anticipation'] == 1
    assert m['clipped_samples'] == 2
    assert m['false_alarm_fraction'] == 1
