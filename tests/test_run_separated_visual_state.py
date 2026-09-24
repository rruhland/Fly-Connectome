"""Matched mixed training corpus keeps held-out shapes unseen."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diverse_visual_experience import TEST_SHAPES, TRAIN_SHAPES
from run_separated_visual_state import make_training_cases


def test_mixed_training_has_clean_interruption_and_noise_without_test_shapes():
    cases = make_training_cases()
    assert len(cases) == 128
    assert {kind for kind, *_ in cases} == {'clean', 'interruption', 'noise'}
    assert {shape for _, shape, _ in cases} == set(TRAIN_SHAPES)
    assert set(TRAIN_SHAPES).isdisjoint(TEST_SHAPES)
    assert all(len(events) == 18 for _, _, events in cases)
