import numpy as np

from scripts.experimental.tt45_tactical_coverage import arc_shape, features
from scripts.experimental.tt45_tactical_goal import Envelope


def test_shot_projection_reflects_lateral_motion_and_rejects_inward_flight():
    p = np.array([[.03, -1.52, 1.04], [-.03, -1.52, 1.04], [0, 0, 1.04]])
    v = np.array([[.25, 3.85, 3.35], [-.25, 3.85, 3.35], [0., -4., 2.]])
    fit = arc_shape(p, v)
    np.testing.assert_allclose(fit[0], fit[1] * [-1, 1, 1, 1])
    assert 0. <= fit[0, 1] < 2.
    assert np.isnan(fit[2]).all()


def test_upstream_goal_is_bounded_and_favors_the_score_earned_pace():
    obj = Envelope(dict(selected=dict(depth=[.34, .76], pace=[2.1, 2.65], center=[0, .52, 2.4])))
    m = np.array([[0., .52, 2.4, 3.], [0., .52, 1.8, 3.], [0., .52, 1.4, 3.], [.55, 1.3, 4.8, 3.]])
    assert obj.contains(m[0]) and not obj.contains(m[1])
    score = obj.penalty(m)
    np.testing.assert_array_equal(np.diff(score) >= 0., np.ones(3, dtype=bool))
    assert score[0] == 0 and score[-1] == 1.35


def test_outgoing_model_need_not_see_any_target_request():
    x = [{"incoming": [0, 1.02, 0, -4, 1.1], "outgoing": [0, .52, 2.4, 3.2]}]
    assert np.isfinite(features(x, ("incoming", "outgoing"))).all()
