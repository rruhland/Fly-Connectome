import numpy as np

from scripts.experimental.tt39_hindsight import DT, camera_height, future_state
from scripts.experimental.tt39_hindsight_prospective import opportunity


def observable(reversal=False):
    p0 = np.array([.07, -.76, .86])
    v0 = np.array([.15, -3.35, 1.10])
    a = np.array([-.04, 1., -10.])
    samples = []
    for k in range(20):
        t = k * DT
        p = p0 + t * v0 + .5 * t**2 * a
        if reversal:
            p[1] = -.79 + .7 * k * DT
        samples.append(dict(x=np.r_[p, v0 + t * a + [0., 0., .24]].tolist(),
                            xyz=p.tolist() if k % 2 == 0 else None))
    return samples, p0, v0


def test_fit_uses_only_later_exposures_and_corrects_causal_velocity():
    frames, p, v = observable()
    fit = future_state(frames, 0)
    assert fit and all(i > 0 for i in fit["future_ticks"])
    np.testing.assert_allclose(fit["reconstructed"][:3], p, atol=.008)
    np.testing.assert_allclose(fit["reconstructed"][3:], v, atol=.07)
    assert fit["change"][-1] < -.17


def test_visible_opposite_direction_is_not_an_incoming_counterfactual():
    frames, *_ = observable(reversal=True)
    assert future_state(frames, 0) is None


def test_optical_height_only_interpolates_a_visible_inbound_crossing():
    raw = [{"xyz": v} for v in ([0., -1.05, .97], [0., -1.15, .99])]
    assert np.isclose(camera_height(raw), .98)
    assert camera_height(raw[::-1]) is None


def test_prospective_first_opportunity_needs_fresh_stereo_before_signalling():
    near = [0., -.8, .9, 0., -3., .4]
    old = [0., -.60, .9, 0., -3., .4]
    h = [dict(x=old, xyz=old[:3]), dict(x=old, xyz=None), dict(x=near, xyz=None),
         dict(x=near, xyz=near[:3]), dict(x=near, xyz=None)]
    assert opportunity(h) == 3
