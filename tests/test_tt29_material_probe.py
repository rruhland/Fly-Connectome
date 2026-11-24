"""Check the observable drop-arc fit and its coverage gate."""

import numpy as np

from scripts.experimental.tt29_material_probe import observed_ratio


def test_two_optical_arcs_measure_normal_rebound():
    time = np.arange(31) / 60.
    impact = .2
    elapsed = time - impact
    vertical_speed = np.where(elapsed < 0., -2., 1.70)
    z = 1.12 + vertical_speed * elapsed - 4.905 * elapsed**2
    samples = np.column_stack((time, np.zeros(len(time)), np.full(len(time), -.70), z))
    measured = observed_ratio(samples)
    np.testing.assert_allclose(measured['optical_restitution'], .85, atol=1e-12)
    np.testing.assert_allclose(measured['impact_time_s'], impact, atol=1e-12)
    assert observed_ratio(samples[(time < .155) | (time > .255)]) is None
