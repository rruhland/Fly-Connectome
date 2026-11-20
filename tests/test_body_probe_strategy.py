"""Exercise phase accounting and source-to-target calibration boundaries."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

import body_probe_strategy as probe
from body_reaching_arena import ReachCase, ReachingArena


def test_reward_assignment_selects_actual_fast_policy():
    costs = np.full((20, len(probe.STRATEGIES)), 20)
    costs[:, 6] = 7
    assert probe.fit_strategy(costs) == probe.STRATEGIES[6]
    shuffled = probe.shuffled_costs(costs, probe.SHUFFLE_SEED)
    assert np.array_equal(np.sort(shuffled, axis=1), np.sort(costs, axis=1))
    assert not np.array_equal(shuffled, costs)


def test_startup_ignores_goal_and_counts_every_command():
    first_case = ReachCase((.4, -.2), (.4, .2))
    second_case = ReachCase((.4, -.2), (.6, .05))
    strategy = probe.Strategy('antithetic', probe.BASELINE_BONUS)
    first = ReachingArena('arm_remapped', first_case, max_steps=14)
    second = ReachingArena('arm_remapped', second_case, max_steps=14)
    for arena in (first, second):
        receipt = probe.run_episode(arena, strategy)
        assert receipt.startup_commands == 4
        assert receipt.total_commands == arena.steps == receipt.startup_commands + receipt.task_commands
        assert receipt.task_commands_changed_by_bonus <= receipt.task_commands
        assert receipt.commands_to_held_goal_or_miss == (
            arena.steps if arena.success else arena.max_steps + 1
        )
    assert first.trace[:5] == second.trace[:5]


def test_episode_has_fresh_body_estimate_and_transient_is_unscored():
    case = ReachCase((.4, -.2), (.4, .2))
    strategy = probe.Strategy('two_axes', probe.BASELINE_BONUS)
    before = probe.run_episode(ReachingArena('arm', case), strategy)
    probe.run_episode(ReachingArena('arm_remapped', case), strategy)
    after = probe.run_episode(ReachingArena('arm', case), strategy)
    assert before == after

    near_case = ReachCase((.4, 0.), (.43, 0.))
    too_short = probe.run_episode(ReachingArena('carriage', near_case, max_steps=3), strategy)
    assert too_short.total_commands == 3
    assert not too_short.held_goal
    assert too_short.commands_to_held_goal_or_miss == 4
