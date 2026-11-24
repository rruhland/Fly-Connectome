"""Terminal shooting uses achieved endpoints and bounded smooth controls."""

import numpy as np

from scripts.experimental.tt39_terminal_servo import desired_pose, edits, endpoint_vector, shot_basis, solve_terminal


def test_schedule_changes_two_time_bases_without_moving_gripper():
    basis = shot_basis(15)
    anchor = np.ones((15, 6))
    changes = np.ones((2, 10))
    sequence = edits(anchor, changes, basis)
    assert sequence.shape == (2, 15, 6)
    np.testing.assert_array_equal(sequence[:, :, -1], 1.)
    assert np.max(sequence[:, 0, :5]) < np.min(sequence[:, -1, :5])


def test_target_normal_is_signed_and_offset_to_the_body_side_of_the_ball():
    p, v = np.array([.06, -1.45, .92]), np.array([.05, -3.55, -.4])
    target = desired_pose(p, v, 3.7)
    assert target[4] > .5
    np.testing.assert_allclose(np.linalg.norm(target[3:6]), 1.)
    np.testing.assert_allclose(p - target[:3], .026 * target[3:6])
    assert 0. < target[6] < 2.


def test_solve_tracks_achieved_terminal_motion_after_actuator_lag():
    class Toy:
        def endpoints(self, desired):
            # Body speed depends on the final commands relative to earlier motion.
            end = (desired[:, -3, :5] * .36 + desired[:, -6, :5] * .64)
            velocity = (desired[:, -2, :3] - desired[:, -6, :3]) * 6.
            n = np.c_[end[:, 3], np.ones(len(end)), end[:, 4]]
            n /= np.linalg.norm(n, axis=1)[:, None]
            return np.c_[end[:, :3], n, velocity]

    body = Toy()
    desired = np.array([.027, .030, -.045, .09, .99, .11, .24])
    desired[3:6] /= np.linalg.norm(desired[3:6])
    schedule, report = solve_terminal(body, np.zeros((15, 6)), desired)
    assert report["weighted_after"] < .38 * report["weighted_before"]
    assert not np.any(schedule[:, 5])
    actual = body.endpoints(schedule[None])
    np.testing.assert_allclose(np.linalg.norm(endpoint_vector(actual, desired)), report["weighted_after"])
