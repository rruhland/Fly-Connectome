import numpy as np

from scripts.experimental import tt38_joint_mpc as mpc
from scripts.experimental.tt41_bounce_joint import (
    commands,
    destination,
    receiving_height,
)


def test_receive_height_is_only_a_causal_camera_gate():
    x = np.array([0.09, -1.13, 1.011, 0.24, -3.33, 0.47])
    dt = (-1.53 - x[1]) / (0.98 * x[4])
    np.testing.assert_allclose(receiving_height(x), x[2] + dt * x[5] - 4.905 * dt**2)
    assert receiving_height(None) is None
    assert receiving_height(x * [1, 1, 1, 1, -1, 1]) is None


def test_forward_and_upward_target_can_clear_the_net():
    p = np.array([-0.14, -1.55, 0.99])
    for pace in (3.12, 3.60, 4.35, 4.86):
        v = destination(p, pace, 0.59)
        time = np.clip(-p[1] / (0.93 * pace), 0.13, 0.62)
        assert p[2] + v[2] * time - 4.905 * time**2 >= 1.015 - 1e-9
        assert v[1] == pace
        assert v[2] > 1.5


def test_face_inverse_issues_references_and_keeps_gripper_unchanged():
    j = np.zeros((3, 5))
    j[:3, :3] = np.eye(3) * 0.22
    angle = np.zeros((3, 5))
    angle[2, 3] = 0.92
    body = (
        0.112,
        np.array([0.03, -1.57, 1.006]),
        np.array([0, 0.96, 0.28]),
        np.array([0.03, -0.01, 0.035]),
        0.028,
        j,
        angle,
        np.array([0.11, 2.51, 1.96]),
        np.zeros(3),
    )
    u = np.array([-0.1, -3.85, -0.42])
    law = [1.88, 0.01]
    seq = np.zeros((mpc.HORIZON, 6))
    seq[:, 5] = -0.1
    solo = list(commands(seq, body, u, "solo", law))
    control = list(commands(seq, body, u, "still", law))
    assert len(solo) == len(control) == 16
    for (one, target), (two, in_control) in zip(solo, control, strict=True):
        assert one.shape == two.shape == seq.shape
        np.testing.assert_allclose(one[:, 5], seq[:, 5])
        np.testing.assert_allclose(two[:, 5], seq[:, 5])
        np.testing.assert_allclose(target, in_control)
    assert any(
        not np.array_equal(a, b) for (a, _), (b, _) in zip(solo, control, strict=True)
    )
