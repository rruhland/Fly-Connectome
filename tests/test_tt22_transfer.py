import numpy as np

from scripts.experimental.tt22_transfer import MotionWeights, ViewMotion, patches


def test_role_free_fit_uses_the_same_geometry_for_shifted_axis_swapped_tracks():
    windows = np.array([[[0., 0.], [2., 1.], [4., 2.], [6., 3.], [8., 4.]],
                        [[1., 3.], [2., 5.], [4., 8.], [7., 12.], [11., 17.]]])
    first, second = MotionWeights(), MotionWeights()
    first.update(*patches(windows))
    second.update(*patches(windows[:, :, ::-1] + 12.))
    assert np.allclose(first.weights, second.weights)
    restored = MotionWeights.restore(first.state())
    assert restored.samples == first.samples
    assert np.array_equal(restored.weights, first.weights)


def test_optical_gap_fill_stops_without_a_fresh_observation():
    optical = ViewMotion((0., 0., 1.))
    for tick in range(4):
        point, imputed = optical.step(tick, (2*tick, 4*tick))
        assert np.array_equal(point, (2*tick, 4*tick)) and not imputed
    prediction, imputed = optical.step(4, None)
    assert np.array_equal(prediction, (8, 16)) and imputed
    assert optical.step(5, None) == (None, False)
