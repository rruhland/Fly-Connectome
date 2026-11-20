"""Verify command rebinding and equal charged probes for the supplemental comparison."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from body_effect_learner import BodyLearner
from body_motor_alignment import BodyMotorAlignment, startup_commands
from body_reaching_arena import ReachCase, TASK_X, TASK_Y, _arm_configuration
from run_body_motor_alignment import GeometryReachingArena, _comparison


def _source_effects(position):
    x, y = position
    return np.array([[.035 + .018 * y, -.011], [.012 + .02 * x, .039 - .016 * y]])


def _prior():
    learner = BodyLearner()
    rng = np.random.default_rng(548)
    for _ in range(300):
        position, command = rng.uniform(-.8, .8, (2, 2))
        learner.reset_episode()
        learner.observe(position, command, position + _source_effects(position) @ command)
    return learner.export_prior()


def test_adapter_recovers_unannounced_binding_and_retains_the_field():
    prior = _prior()
    learner = BodyMotorAlignment(prior)
    rng = np.random.default_rng(182)
    actual_adapter = np.array([[0., -1.], [1., 0.]])
    for _ in range(8):
        for command in startup_commands(rng):
            position = rng.uniform(-.8, .8, 2)
            learner.reset_episode()
            next_position = position + _source_effects(position) @ actual_adapter @ command
            learner.observe(position, command, next_position)
    assert learner.updates == 32
    assert learner.source.export_prior() == prior
    np.testing.assert_allclose(learner.adapter, actual_adapter, atol=.05)

    learner.reset_episode()
    new_position = np.array([-.73, .76])
    command = np.array([-.52, .69])
    predicted = learner.predict_displacement(new_position, command)
    expected = _source_effects(new_position) @ actual_adapter @ command
    np.testing.assert_allclose(predicted, expected, atol=.002)


def test_all_comparators_receive_the_whole_balanced_probe_prefix():
    case = ReachCase((.42, -.25), (.56, .3))
    probes = startup_commands(np.random.default_rng(632))
    np.testing.assert_array_equal(probes[:2], -probes[2:])
    np.testing.assert_array_equal(np.abs(probes).sum(axis=0), [1.8, 1.8])
    rows = _comparison('arm_geometry', case, _prior(), probes, 632, 0)
    assert len(rows) == 4
    for row in rows.values():
        assert row['commands'] >= len(probes)
        assert row['probe_path'] == rows['aligned']['probe_path']
    assert rows['unchanged']['updates'] == 0
    for name in ('aligned', 'fresh', 'local_alignment'):
        assert rows[name]['commands'] == rows[name]['updates']


def test_geometry_variant_starts_in_the_shared_reachable_workspace():
    points = [(x, y) for x in (*TASK_X, .43) for y in (*TASK_Y, 0.)]
    for position in points:
        case = ReachCase(position, (.38, .13))
        arena = GeometryReachingArena(case)
        np.testing.assert_allclose(arena.ground_truth_position, position, atol=1e-12)
        config = _arm_configuration(position, arena._spec.links)
        assert all(lo < val < hi for (lo, hi), val in zip(arena._spec.limits, config))
