"""Exercise visual command learning and carry without an arena dependency."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from body_effect_learner import BodyLearner


def _effects(position):
    x, y = position
    return np.array([[.039 + .018 * y, -.013], [.017 + .02 * x, .036 - .015 * y]])


def _learned_prior():
    rng = np.random.default_rng(153)
    learner = BodyLearner()
    for _ in range(300):
        position = rng.uniform(-.65, .65, 2)
        command = rng.uniform(-1., 1., 2)
        learner.reset_episode()
        learner.observe(position, command, position + _effects(position) @ command)
    return json.loads(json.dumps(learner.export_prior()))


def test_carried_effects_predict_unseen_commands_and_freeze_weights():
    prior = _learned_prior()
    frozen = BodyLearner(prior=prior, updates_enabled=False)
    cold = BodyLearner(updates_enabled=False)
    position = np.array([.24, -.31])
    command = np.array([-.68, .57])
    target = position + np.array([.25, -.25])
    expected = _effects(position) @ command
    np.testing.assert_allclose(frozen.predict_displacement(position, command), expected, atol=.002)
    np.testing.assert_array_equal(cold.act(position, target), [0., 0.])
    assert np.linalg.norm(frozen.act(position, target)) > .5

    frozen.observe(position, command, position - expected)
    frozen.reset_episode()
    np.testing.assert_allclose(frozen.predict_displacement(position, command), expected, atol=.002)
    assert frozen.updates == 0
    assert frozen.export_prior() == prior


def test_adaptive_carry_relearns_remapped_commands_from_visual_feedback():
    prior = _learned_prior()
    learner = BodyLearner(seed=94, prior=prior)
    position = np.array([.38, -.25])
    target = np.array([-.18, .3])
    mapping = np.array([[0., -1.], [1., 0.]])
    command = np.array([1., 0.])
    actual_effect = _effects(position) @ mapping @ command
    assert np.linalg.norm(learner.predict_displacement(position, command) - actual_effect) > .035

    for _ in range(80):
        command = learner.act(position, target)
        assert np.max(np.abs(command)) <= 1.
        after = position + _effects(position) @ mapping @ command
        learner.observe(position, command, after)
        position = after

    assert np.linalg.norm(position - target) < .025
    assert learner.updates == 80
    assert learner.export_prior() != prior


def test_episode_reset_clears_motion_without_changing_weights():
    prior = _learned_prior()
    # Give the serialized model a known image-displacement response.
    prior['weights'][-2:] = [[.025, 0.], [0., .025]]
    learner = BodyLearner(prior=prior, updates_enabled=False)
    learner.observe([0., 0.], [0., 0.], [.04, -.02])
    np.testing.assert_allclose(learner.predict_displacement([.04, -.02], [0., 0.]), [.02, -.01])
    learner.reset_episode()
    np.testing.assert_array_equal(learner.predict_displacement([.04, -.02], [0., 0.]), [0., 0.])
    assert learner.export_prior() == prior
