import numpy as np

from scripts.experimental.tt27_context_memory import Memory
from scripts.experimental.tt27_miss_learning import OpticalError
from scripts.experimental.tt27_optical_dynamics import MaterialMemory, observed_bounce, predict


def test_observed_bounce_recovers_change_from_camera_positions():
    gravity = 9.81
    downward = 2.0
    rebound = 0.67
    forward_ratio = 0.80
    trail = []
    for tick in range(15):
        t = (tick - 7) / 30
        vz = -downward if t <= 0 else downward * rebound
        yvel = -3.0 if t <= 0 else -3.0 * forward_ratio
        trail.append([tick, 0.12 + 0.1 * t, -0.30 + yvel * t, 0.021 + vz * t - 0.5 * gravity * t * t])
    np.testing.assert_allclose(observed_bounce(trail), [rebound, forward_ratio, 0.0], atol=1e-6)
    material = MaterialMemory([1, 1, 0])
    material.observe(trail)
    predicted = predict(trail[:5], -0.50, material.current)
    dt = 0.20 / (3.0 * forward_ratio)
    np.testing.assert_allclose(
        predicted[:2], [0.12 + 0.1 * dt, 0.021 + downward * rebound * dt - 0.5 * gravity * dt**2]
    )


def test_local_reward_memory_only_changes_from_the_action_played():
    m = Memory([], context=True, local=True)
    reading = np.array([0.8, 0.85])
    assert m.choose(reading, 4) == 0
    for _ in range(5):
        m.add(reading, 4, 1, 1)
        m.add(reading, 4, 0, 0)
    assert m.choose(reading, 4) == 1


def test_observed_crossing_interpolates_motion_and_declines_a_long_gap():
    a = np.array([8, 0.1, 0.15, 0.2])
    b = np.array([9, 0.22, -0.05, 0.1])
    np.testing.assert_allclose(OpticalError.crossing([a, b]), [0.19, 0.125])
    b[0] = 19
    assert OpticalError.crossing([a, b]) is None


def test_end_aligned_opponent_moves_only_the_downcourt_coordinate():
    from scripts.experimental.tt22_opponent import RallyOpponent

    rival = RallyOpponent.__new__(RallyOpponent)
    rival.target_profile = (0.12, -0.9, 0.2, 2.4)
    rival.receiving_depth_shift = 0.0
    np.testing.assert_array_equal(rival.target(), rival.target_profile)
    rival.receiving_depth_shift = 0.57
    np.testing.assert_allclose(rival.target(), (0.12, -1.47, 0.2, 2.4))
