import numpy as np

from scripts.experimental.tt43_outgoing_generation import choose, retarget


def test_generator_preserves_grasp_and_shapes_motion_in_current_sensor_time():
    base = np.zeros((21, 6))
    noise = np.ones(10) * .10
    earlier = retarget(base, noise, .11)
    later = retarget(base, noise, .23)
    np.testing.assert_equal(later[:, 5], 0.)
    assert earlier[5, 0] > later[5, 0] >= 0.
    np.testing.assert_allclose(later[-1, :5], .20)


def test_a_consensus_improvement_cannot_trade_away_current_return():
    reference = np.array([.15, .13, .16])
    risk = np.array([[.10, .09, .11], [.06, .06, .06],
                     [.18, .04, .04], [0., 0., 0.], [.08, .08, .08]])
    quality = np.array([-2.56, -2.3, -2.6, -2.6, -2.6])
    robust = np.array([2, 2, 2, 1, 2])
    supported = np.array([True, True, True, True, False])
    assert choose(quality, robust, risk, supported, -2.7, reference) == 0
