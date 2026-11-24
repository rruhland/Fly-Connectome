import numpy as np

from scripts.experimental.tt39_action_value import ActionValue
from scripts.experimental.tt39_centered_joint import adjustment
from scripts.experimental.tt39_turn_time import turn_time


def test_reconstruct_delayed_camera_reversal_from_unambiguous_segments():
    rows = []
    for t in range(48):
        # The tracker continues extrapolating on the incoming arc for four ticks.
        dt = (t - 21) / 60
        velocity = np.array([0.15, -3.0, 0.1]) if t < 25 else np.array([-0.3, 3.5, 2.7])
        position = (
            np.array([0.2, -1.55, 1.1])
            + dt * velocity
            + np.array([0, 0, -4.905]) * dt**2
        )
        rows.append((t, np.r_[position, velocity + np.array([0, 0, -9.81]) * dt]))
    assert abs(turn_time(rows) - 21) < 0.75


def test_direct_preference_records_actually_issued_missed_and_scored_controls_once():
    class Base:
        def get(self, optical, q, reference=None):
            return np.r_[q[:5], -0.1], 1.0

    x = np.array([0.1, -1.0, 1.0, 0.05, -2.0, 0.1])
    qs = np.zeros(6)
    samples = np.repeat(np.r_[x, qs[:5]][None, :], 13, axis=0)
    player = ActionValue(Base(), (samples, np.zeros((13, 5)), np.zeros(13)))
    for leg in range(2):
        for t in range(12):
            player.record_command(t, x, qs, np.ones(6) * 0.1)
        if leg == 0:
            player.score_feedback(1)
            player.score_feedback(1)
        else:
            player.end_point(terminal=True)
    assert len(player.added) == 12
    assert sum(r[2] for r in player.added) == 6


def test_interior_slide_preserves_virtual_surface_normal_and_gripper():
    class Body:
        def pose(self, q, *, jacobian):
            return (
                np.zeros(3),
                np.array([0.0, 1.0, 0.0]),
                np.c_[np.eye(3), np.zeros((3, 2))],
                np.zeros((3, 5)),
            )

    packet = np.zeros(19)
    packet[0] = 0.12
    packet[1:4] = [0.065, 0.021, 0]
    packet[18] = 1
    slide, rotation, velocity = adjustment(
        Body(), packet, np.zeros(9), np.array([0, -3.0, 0]), 21
    )
    assert slide[-1, 0] > 0.06 and np.count_nonzero(slide[:, 1:]) == 0
    assert np.all(rotation[:, -1] == 0) and np.all(velocity[:, -1] == 0)
