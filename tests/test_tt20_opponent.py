import numpy as np

from scripts.experimental.tt4_arena import _arm_angles
from scripts.experimental.tt19_so101 import SO101Arena
from scripts.experimental.tt20_opponent import aim_angles, install_opponent


def test_vectorized_inverse_kinematics_matches_rival_base_coordinates():
    base = 1.435
    xyz = [(0., .90, .95), (.14, .85, 1.11), (-.20, .95, 1.03)]
    x, y, z = np.array(xyz).T
    actual = aim_angles(base, x, y, z, .24)
    expected = [_arm_angles(left, depth + 1.94 - base, height, -1, .24) for left, depth, height in xyz]
    assert np.allclose(actual, expected)


def test_counterfactual_swing_search_leaves_live_solver_and_score_untouched():
    arena = SO101Arena(venue='studio', detail='preview', strict_endline=True)
    try:
        arena.reset(seed=42, server=-1, render=False)
        rival = install_opponent(arena)
        arena.ball[:] = (.04, .20, 1.07)
        arena.velocity[:] = (.13, 3., -.03)
        arena._last_hitter = 1
        arrays = (arena.data.qpos, arena.data.qvel, arena.data.ctrl, arena.data.xfrc_applied)
        before = [value.copy() for value in arrays]
        events, result, clock = list(arena.events), arena.point_result(), arena.data.time
        tape, evidence = rival.plan()
        assert evidence['candidates'] > 0
        assert all(np.array_equal(old, new) for old, new in zip(before, arrays, strict=True))
        assert arena.events == events and arena.point_result() == result and arena.data.time == clock
        if tape is not None:
            assert tape.shape[1] == 4 and np.max(np.abs(tape)) <= 1.
    finally:
        arena.close()
