"""Check the swept racket contact used by TT19 shooting."""
import numpy as np

from scripts.experimental.tt19_shoot import HORIZON, _surface


def test_search_distinguishes_face_rim_and_miss():
    poses = np.zeros((3, HORIZON + 1, 5))
    poses[:, :, 1:3] = [-.91, .20]
    poses[1, :, 0] = -.085
    poses[2, :, 0] = -.125
    ball = np.zeros((HORIZON + 1, 3))
    ball[:, 1] = np.linspace(-.72, -1.18, HORIZON + 1)
    ball[:, 2] = .20
    hit, eta, _, _, normal, _, _, radial, _ = _surface(poses, ball, np.gradient(ball, axis=0))
    assert hit.tolist() == [True, True, False]
    assert eta[0] <= eta[1]
    assert radial[1] > .076
    assert normal[1, 0] > .15


def test_moving_face_can_reach_descending_ball_before_it_passes():
    poses = np.zeros((2, HORIZON + 1, 5))
    poses[:, :, 1] = -.99
    poses[1, :, 1] += np.minimum(np.arange(HORIZON + 1) * .035, .13)
    poses[:, :, 2] = .24
    ball = np.zeros((HORIZON + 1, 3))
    ball[:, 1] = -.86
    ball[:, 2] = np.linspace(.45, -.15, HORIZON + 1)
    hit, *_ = _surface(poses, ball, np.gradient(ball, axis=0))
    assert hit.tolist() == [False, True]
