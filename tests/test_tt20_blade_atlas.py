"""Check the factory dry-state blade pose supplied to the rebuilt shortlist."""

import numpy as np

from scripts.experimental.tt4_arena import TABLE_HEIGHT
from scripts.experimental.tt20_blade_atlas import blade_pose


def test_blade_pose_applies_the_dry_quaternion_and_faces_the_court():
    angle = np.pi / 6
    pose = np.array([.2, -.7, TABLE_HEIGHT + .1, np.cos(angle / 2), 0., np.sin(angle / 2), 0.])
    result = blade_pose(pose, np.array([.03, 0., 0.]), np.array([0., -.8, -.6]))
    np.testing.assert_allclose(result, [.2 + .03 * np.cos(angle), -.7, .1 - .03 * np.sin(angle),
                                       .6 * np.sin(angle), .6 * np.cos(angle)])
