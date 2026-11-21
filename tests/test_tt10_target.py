import numpy as np
import pytest

pytest.importorskip('mujoco')

from scripts.experimental.tt10_target import BLOCK, PULSE, CursorPlayer, TargetArena, read_markers
from scripts.experimental.tt10_target_remap import MatchedMotorPlayer


def test_reaching_scores_only_after_two_ticks_and_once_for_a_target():
    arena = TargetArena(seed=7, body='offset')
    try:
        arena.reset(seed=7)
        arena.set_target(arena.paddle.copy())
        cursor, goal = read_markers(arena.frame())
        np.testing.assert_array_equal(cursor, goal)
        assert [arena.step(np.zeros(5), render=False)[1] for _ in range(BLOCK)] == [0, 1] + [0] * (BLOCK - 2)
        arena.set_target(arena.paddle.copy())
        assert [arena.step(np.zeros(5), render=False)[1] for _ in range(2)] == [0, 1]
    finally:
        arena.close()


def test_fixed_motor_map_still_counts_probes_and_respects_range():
    arena = TargetArena(seed=8, body='offset')
    try:
        arena.reset(seed=8)
        arena.set_target(arena.paddle + (.16, .10, .08))
        rgb = arena.frame()
        controller = CursorPlayer(np.zeros((3, 5)), update=False, exploration='evidence')
        responses = [controller.act(rgb).copy() for _ in range(6)]
        assert len({np.argmax(abs(a)) for a in responses[:5]}) == 5
        assert np.max(np.abs(responses)) <= 1.
        np.testing.assert_allclose(PULSE * np.sum(responses, axis=0), controller.requested)
        np.testing.assert_equal(controller.cross, np.zeros((5, 3)))
    finally:
        arena.close()


def test_familiar_directions_identified_without_access_to_channel_names():
    directions = np.diag([.10, .06, .03])
    player = MatchedMotorPlayer(directions)
    player.measured[0] = -directions[:, 2] + .001
    player.measured[1] = directions[:, 0] + .001
    projected = player.projected_map()
    np.testing.assert_array_equal(projected[:, 0], -directions[:, 2])
    np.testing.assert_array_equal(projected[:, 1], directions[:, 0])
    np.testing.assert_array_equal(projected[:, 2], np.zeros(3))
