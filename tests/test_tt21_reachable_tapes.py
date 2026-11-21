import numpy as np

from scripts.experimental.tt19_model_rollout import reachable_tapes
from scripts.experimental.tt19_shoot import HORIZON, _surface


def test_sweep_retains_midframe_contact_and_original_indices():
    points = np.zeros((HORIZON + 1, 3))
    points[:, 1] = -0.91
    points[:, 2] = 0.14
    speeds = np.zeros_like(points)
    speeds[:, 1] = -0.06
    poses = np.zeros((3, HORIZON + 1, 5))
    poses[:, :, :3] = points
    poses[:, :, 0] = 0.3
    poses[1, 1:, 0] = -0.3
    poses[1, :, 1] -= 0.02

    hit, eta, *_ = _surface(poses, points, speeds)
    np.testing.assert_array_equal(hit, [False, True, False])
    assert 0 < eta[1] < 1
    np.testing.assert_array_equal(reachable_tapes(poses, points), [1])

    points[:, 1] = -0.5
    assert reachable_tapes(poses, points).size == 0
