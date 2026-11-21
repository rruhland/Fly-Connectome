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


def test_online_lane_label_comes_from_later_incoming_images():
    from scripts.experimental.tt21_ready_position_match import new_observation

    choice = {"tick": 40, "cue": [1.0, 0.12, 0.02]}
    row = {
        "seed": 18,
        "decision": choice,
        "optical": [
            [20, [0.3, -0.55, 0.15]],
            [40, [0.13, 0.3, 0.18]],
            [47, [0.18, -0.35, 0.13]],
            [48, [0.19, -0.43, 0.12]],
            [49, [0.20, -0.51, 0.10]],
        ],
    }
    assert new_observation(dict(row, optical=row["optical"][:-1])) is None
    actual = new_observation(row)
    assert actual["observed_tick"] == 49
    assert actual["cue"] == choice["cue"]
    np.testing.assert_allclose(actual["next_x"], 0.24625)
