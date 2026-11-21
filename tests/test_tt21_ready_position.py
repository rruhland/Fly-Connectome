import numpy as np

from scripts.experimental.tt21_ready_position import (
    inbound_features,
    outbound_features,
    rotated_poses,
)


def test_direction_features_require_consecutive_motion_in_the_correct_direction():
    outgoing = [(i + 30, [0.1 + 0.01 * i, 0.27 + 0.1 * i, 0.15]) for i in range(3)]
    incoming = [(i + 41, [0.14 + 0.01 * i, 0.59 - 0.1 * i, 0.19]) for i in range(3)]

    assert inbound_features(outgoing) is None
    assert outbound_features(incoming) is None
    np.testing.assert_allclose(outbound_features(outgoing), [1, 0.145, 0.2])
    np.testing.assert_allclose(inbound_features(incoming), [1, 0.199, 0.1])
    assert inbound_features([(29, incoming[0][1]), *incoming[1:]]) is None


def test_catalog_rotation_occurs_about_the_shoulder_not_the_table_origin():
    poses = np.array([[[0.0, -0.75, 0.17, 0.0, 0.1]]])
    shifted = rotated_poses(poses, [0.0, -1.0], -np.pi / 4)
    np.testing.assert_allclose(shifted[0, 0, :3], [np.sqrt(0.5) / 4, -1.0 + np.sqrt(0.5) / 4, 0.17])
    assert shifted[0, 0, 3] > 0.0
    np.testing.assert_array_equal(poses, [[[0.0, -0.75, 0.17, 0.0, 0.1]]])
