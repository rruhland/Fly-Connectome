"""Check privileged feasibility and straight action replay on the fixed model."""

import numpy as np
import pytest

pytest.importorskip('mujoco')
from scripts.experimental.tt24_bounce import BounceArena
from scripts.experimental.tt24_bounce_ceiling import PrivilegedCeiling


def test_privileged_commands_leave_integration_to_the_standard_arena():
    arena = BounceArena(view='none')
    try:
        arena.reset(seed=243001, split='held')
        controller = PrivilegedCeiling(arena)
        commands = []
        for _ in range(arena.max_steps):
            qpos, qvel, controls = (state.copy() for state in (arena._data.qpos, arena._data.qvel, arena._data.ctrl))
            action = controller.command()
            np.testing.assert_array_equal(arena._data.qpos, qpos)
            np.testing.assert_array_equal(arena._data.qvel, qvel)
            np.testing.assert_array_equal(arena._data.ctrl, controls)
            assert np.abs(action).max() <= controller.settings.rate_limit < 1.
            commands.append(action)
            _, _, done = arena.step(action)
            if done:
                break
        expected = arena.judge_metrics()
        assert expected['termination'] == 'timeout'
        assert expected['longest_consecutive_rebounds'] >= 10

        arena.reset(seed=243001, split='held')
        for action in commands:
            _, _, done = arena.step(action)
        assert done
        assert arena.judge_metrics() == expected
    finally:
        arena.close()
