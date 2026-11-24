import numpy as np

from scripts.experimental.tt33_joint import JointPolicy
from scripts.experimental.tt36_composed import position_preserving
from scripts.experimental.tt36_gate import Gate


def test_direct_joint_offset_respects_slew_and_leaves_ready_pose_unchanged():
    ready = np.zeros(6)
    model = {
        "ready": ready.tolist(),
        "weights": np.zeros((1, 6)).tolist(),
        "centers": np.zeros((1, 11)).tolist(),
        "options": {
            "features": "raw-proprioception",
            "width": 1.0,
            "feedback": 0.0,
            "limit": 0.8,
            "recovery": 0.4,
        },
    }
    policy = JointPolicy(model, np.full(6, 0.1), np.full(6, -2.0), np.full(6, 2.0))
    raw = np.array([0.0, -0.65, 1.1, 0.0, -3.6, 0.0])
    np.testing.assert_allclose(
        policy.command(raw, ready, offset=np.array([0.12, -0.02, 0.0, 0.0, 0.0])),
        [0.8, -0.2, 0.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_array_equal(policy.home, ready)
    np.testing.assert_allclose(
        policy.command(None, ready, offset=np.ones(5)), [-0.4, 0.2, 0.0, 0.0, 0.0, 0.0]
    )


def test_gate_needs_local_evidence_and_keeps_its_first_choice():
    sample = np.zeros(11)
    ck = {
        "observations": [sample.tolist()],
        "scale": np.ones(11).tolist(),
        "advantages": [[0, -1, 2, 1]],
    }
    gate = Gate(ck)
    assert gate.choose(None, np.zeros(6)) == "fixed"
    assert gate.choose(sample[:6], np.zeros(6)) == "hit"
    assert gate.choose(np.ones(6) * 20, np.ones(6) * 20) == "hit"
    assert Gate(ck).choose(np.ones(6) * 20, np.ones(6) * 20) == "fixed"


def test_late_command_preserves_unconstrained_joints_and_suppresses_translation():
    body = np.eye(3, 5)
    original = np.array([0.2, 0.3, -0.1, 0.4, -0.5])
    residual = position_preserving(original, body)
    assert np.linalg.norm(body @ residual) < 0.001 * np.linalg.norm(body @ original)
    np.testing.assert_array_equal(residual[3:], original[3:])
    np.testing.assert_allclose(
        position_preserving(original, np.zeros_like(body)), original
    )
