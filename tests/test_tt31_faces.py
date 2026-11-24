import numpy as np

from scripts.experimental.tt31_faces import (
    READY,
    ROLL,
    SETTINGS,
    FaceMemory,
    FaceTwin,
    SignedBladeIK,
)


def test_signed_ik_preserves_both_reachable_orientations():
    arena = FaceTwin(**SETTINGS)
    point = np.array([.23, -1.315, 1.06])
    desired = np.array([0., .98, .199])
    desired /= np.linalg.norm(desired)
    starts = []
    try:
        for sign in (1, -1):
            ik = SignedBladeIK(arena, sign)
            q, (center, face, *_) = ik.fit(point, desired, np.array(READY[:5]))
            assert np.linalg.norm(center - point) < .016
            assert face @ desired > .96
            assert ik.pose(q)[1] @ face > .999
            physical = ik.data.xmat[ik.gripper].reshape(3, 3) @ ik.normal
            assert physical @ desired * sign > .96
            starts.append(q)
    finally:
        arena.close()
    assert abs(starts[0][ROLL] - starts[1][ROLL]) > 2.7


def test_rotated_recall_is_only_a_candidate_and_keeps_the_archive():
    memory = FaceMemory((-2.74, 2.84), 60)
    key = np.zeros(5)
    goals = np.zeros((41, 6))
    goals[:27, ROLL] = -2.4
    goals[27:, ROLL] = READY[ROLL]
    memory.add(key, goals, .35)
    original, alternate = memory.nearest(key, 1)
    np.testing.assert_array_equal(original.targets, goals)
    np.testing.assert_allclose(alternate.targets[:, ROLL], -2.4 + np.pi)
    np.testing.assert_array_equal(alternate.targets[:, :ROLL], goals[:, :ROLL])
    assert len(memory.examples) == 1
