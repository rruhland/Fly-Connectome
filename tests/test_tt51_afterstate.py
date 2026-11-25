import numpy as np

from scripts.experimental.tt51_afterstate import projected, variance


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
