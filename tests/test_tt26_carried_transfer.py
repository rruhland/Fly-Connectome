import numpy as np

from scripts.experimental.tt26_carried_player import CarriedPlayer
from scripts.experimental.tt26_exploration_transfer import decisions
from scripts.experimental.tt26_visual_acquisition import VisualRewardMemory


def test_local_receiving_reward_survives_reload_with_shared_tool(tmp_path):
    goals = np.arange(30, dtype=float).reshape(6, 5) / 40
    offset = np.array([0.0, 0.0, 0.0, 0.0, 0.24])
    player = CarriedPlayer(offset, goals)
    view = np.zeros(14)
    alternative = np.full(14, 170.0)
    for _ in range(5):
        player.receiving.update(view, 2, 1)
        player.receiving.update(view, 1, 0)
        player.receiving.update(alternative, 1, 1)
    path = tmp_path / "player.npz"
    player.save(path)
    resumed = CarriedPlayer.load(path)
    for key, expected in ((view, 2), (alternative, 1)):
        choice, goal = resumed.select(key, np.arange(6), False)
        assert choice == expected
        np.testing.assert_array_equal(goal, goals[expected] + offset)
    resumed.receiving.update(alternative, 4, 0)
    np.testing.assert_array_equal(resumed.adjustment, offset)


def test_anonymous_movements_have_no_initial_visual_assignment():
    learner = VisualRewardMemory(3)
    order = np.array([2, 0, 1])
    assert learner.choose(np.ones(14), order, True) == 2


def test_exploration_uses_only_outcomes_of_queried_joints():
    scores = np.array([2, 0, 0, 5, 1, 2, 3, 2, 0, 4, 1])
    order = [4, 0, 1, 2, 3]
    assert decisions(scores, order) == [0, 0, 6, 6, 3, 3]
