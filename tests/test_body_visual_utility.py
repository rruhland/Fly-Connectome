"""Verify the visual scorer, scalar interface, and checkpoint boundary."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from body_visual_utility import (
    ChoiceArena, Layout, PayoffRule, UtilityLearner, offset_features,
)
from body_reaching_arena import ReachingArena


def test_camera_offsets_are_translation_invariant_and_choice_scores_swap():
    candidates = np.array(((.4, .1), (.2, -.3)))
    reference = np.array((.2, .2))
    translated = np.array((.3, -.4))
    features = offset_features(candidates, reference)
    assert np.allclose(features, offset_features(candidates + translated, reference + translated))
    utility = UtilityLearner()
    utility.observe(features[0] - features[1], 1)
    assert utility.choose(features, 1) == 0
    assert utility.choose(features[::-1], 0) == 1


def test_checkpoint_restores_scores_with_no_source_precision():
    rng = np.random.default_rng(761)
    utility = UtilityLearner()
    for i in range(8):
        utility.observe(rng.normal(size=5), (-1) ** i)
    checkpoint = utility.checkpoint()
    restored = UtilityLearner(checkpoint, updates_enabled=False)
    assert set(checkpoint) == {'schema', 'weights'}
    assert len(checkpoint['weights']) == 5
    assert np.array_equal(restored.weights, utility.weights)
    assert np.array_equal(restored.precision, UtilityLearner().precision)
    restored.observe(rng.normal(size=5), 1)
    assert restored.checkpoint() == checkpoint


def test_hidden_rule_and_reversal_do_not_change_pixels_or_initial_reward():
    layout = Layout((.3, -.3), ((.4, .10), (.6, .28)), (.7, -.2))
    rule = PayoffRule('quadratic', .3, (1., -.6))
    normal = ChoiceArena('carriage', layout, rule, False, ReachingArena)
    reversed_arena = ChoiceArena('carriage', layout, rule, True, ReachingArena)
    assert np.array_equal(normal.frame(), reversed_arena.frame())
    frame, reward = normal.step((1., 0.))
    reversed_frame, reversed_reward = reversed_arena.step((1., 0.))
    assert reward == reversed_reward == 0
    assert np.array_equal(frame, reversed_frame)


def test_degree_two_scorer_fits_rotated_quadratic_preference():
    rng = np.random.default_rng(6721)
    rule = PayoffRule('quadratic', 1.1, (1.2, -.5))
    learner = UtilityLearner()
    correct = 0
    for i in range(140):
        candidates = rng.uniform(-.6, .6, size=(2, 2))
        reference = rng.uniform(-.4, .4, size=2)
        expected = rule.winner(candidates, reference, False)
        features = offset_features(candidates, reference)
        if i < 80:
            learner.observe(features[0] - features[1], 1 if expected == 0 else -1)
        else:
            correct += learner.choose(features, 0) == expected
    assert correct >= 48
