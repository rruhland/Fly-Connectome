import numpy as np

from scripts.experimental.tt51_afterstate import bootstrap_difference, projected, variance
from scripts.experimental.tt51_afterstate_native import agrees


def test_projection_uses_current_optical_direction_and_gravity():
    p=np.array([.18,-1.10,1.0,.3,-4.3,.85])
    y=projected(p)
    t=.43/(.98*4.3)
    np.testing.assert_allclose(y[:2],[1.+t*.85-4.905*t*t,.18+t*.3])
    assert y[2]==4.3


def test_variance_separates_stroke_outcome_from_independent_draw_noise():
    repeated=[[[1.08,0.,.07]]*6,[[.97,1.,.85]]*6]
    result=variance(repeated)
    np.testing.assert_allclose(result["draw_intraclass"],[1,1,1])
    same=[[ [.97,1,.85],[1.08,0,.07] ]*3]*4
    result=variance(same)
    np.testing.assert_allclose(result["draw_intraclass"],[0,0,0])


def test_interval_is_paired_by_release_and_allows_release_repetition():
    loss=np.array([[.23,.30],[.17,.13],[.20,.85]])
    reference=loss+np.array([.05,.1])
    np.testing.assert_allclose(bootstrap_difference(loss,reference,[3,3,4]),
                               [[-.05,-.1],[-.05,-.1]])


def test_redraw_verification_allows_only_roundoff_and_identical_events():
    a={"step":294,"u":[.992384,1.12]}
    assert agrees(a,{"step":294,"u":[.992384+2e-10,1.12]})
    assert not agrees(a,{"step":294,"u":[.992384+2e-7,1.12]})
    assert not agrees(a,{"step":293,"u":[.992384+2e-10,1.12]})
