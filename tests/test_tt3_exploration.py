"""Check excitation and the early probe release at the real learner entry point."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experimental.tt3_exploration import DWELLS, PROBE_FRAMES, probe_actions
from scripts.experimental.tt3_learning import TableLearner
from scripts.experimental.tt3_visual import Binding, RoleModel, Track


def test_dwell_changes_timing_with_equal_energy_and_full_rank():
    for dwell in DWELLS:
        actions = probe_actions('prior_dwell', 71, learned_dwell=dwell)
        assert actions.shape == (PROBE_FRAMES, 3)
        np.testing.assert_allclose(actions.mean(axis=0), 0., atol=1e-12)
        np.testing.assert_allclose(actions.T @ actions, PROBE_FRAMES / 3 * np.eye(3), atol=1e-12)
        held = np.repeat(actions[::dwell], dwell, axis=0)
        np.testing.assert_array_equal(actions, held)


def test_probe_limit_updates_before_steering_and_resets_each_point():
    role_model = RoleModel((0., 1., 0.), (1., 0., 0., 0., 0.), .1, 1, 1)
    visits = []

    def stop(learner, binding):
        visits.append(learner.frame)
        return learner.frame == 3 and binding.controlled.observed

    learner = TableLearner(role_model, probe_frames=PROBE_FRAMES, stop_probing=stop)
    controlled = Track(7, (0., 0., 0.), (0., 0., 0.))
    relevant = Track(11, (.2, .1, .1), (-.04, 0., 0.))
    learner.binder.update = lambda *_: Binding(controlled, relevant, (controlled, relevant))
    learner.reset(option=0)
    for _ in range(6):
        learner.act(None)
    assert learner.probe_count == 3
    assert visits == [0, 1, 2, 3]
    learner.reset(option=0)
    learner.act(None)
    assert learner.probe_count == 1
    assert visits[-1] == 0
