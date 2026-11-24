"""Check the causal labels and forecast score in the RGB flight assay."""

import numpy as np

from scripts.experimental.tt29_flight_residual import FlightCorrection, observed_crossing, score


def test_late_stereo_crossing_recovers_flight_tangent_at_fixed_depth():
    t = np.arange(7) * .03
    x, y, z = .10 + .20 * t, -.65 - 4. * t, 1.10 - .30 * t - .20 * t**2
    rows = np.stack((t, x, y, z), axis=1)
    crossing_t = .075
    expected = (.10 + .20 * crossing_t,
                1.10 - .30 * crossing_t - .20 * crossing_t**2,
                crossing_t + .06, .20, -4., -.30 - .40 * crossing_t)
    np.testing.assert_allclose(observed_crossing(rows, -.06), expected, atol=1e-12)
    assert observed_crossing(rows, .04) is None


def test_completed_rgb_rows_correct_held_predictions_without_refitting():
    learner = FlightCorrection()
    source = dict(outcome='paired', features=[1., 0., 0., 0., 0., 0.],
                  forecast=[0., 1., .15, 0., -5., .10],
                  observed=[0., 1.03, .15, 0., -5., .10])
    for _ in range(12):
        learner.learn([source])
    parameters = learner.coefficients.copy()
    held = dict(source, forecast=[0., 1.10, .15, 0., -5., .10],
                observed=[0., 1.13, .15, 0., -5., .10])
    result = score([held], learner)
    assert result['corrected_mae']['z_m'] < .01
    assert result['corrected_mae']['z_m'] < result['nominal_mae']['z_m']
    np.testing.assert_array_equal(learner.coefficients, parameters)
