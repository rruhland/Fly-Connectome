import numpy as np

from scripts.experimental import tt76_learned_impulse as motor


def test_no_learning_from_a_tape_without_outgoing_observation():
    rows = [(t, np.array([0, -1, 1, 0, -2, 0])) for t in range(26)]
    assert motor.observe_departure(rows) is None


def test_symmetric_contact_pairs_drive_only_attained_directions():
    basis = np.zeros((3, 21, 6))
    basis[0, :, 1] = 0.10
    basis[1, :, 2] = 0.10
    basis[2, :, 3] = 0.10
    probes = np.zeros((3, 2, 9))
    probes[:, :, 1] = 1
    probes[0, :, 2] = [1, -1]
    probes[1, :, 3] = [1, -1]
    probes[2, 0, 1] = 0
    probes[2, :, 4] = [1, -1]
    changes = motor.velocity_step(np.zeros(3), basis, probes, [0, 2, 4])
    for delta in changes:
        assert np.all(delta[:, 2] > 0)
        assert np.all(delta[:, [0, 1, 3, 4, 5]] == 0)
    probes[:, 0, 1] = 0
    assert motor.velocity_step(np.zeros(3), basis, probes, [1, 0, 0]) == []


def test_pose_and_velocity_profiles_are_independent_and_bounded():
    profiles = motor.profiles(13)
    assert profiles.shape == (2, motor.core.HORIZON)
    assert np.linalg.matrix_rank(profiles) == 2
    assert np.min(profiles[1]) < 0 < np.max(profiles[1])
