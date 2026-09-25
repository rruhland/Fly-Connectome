"""Balanced opt-in exposure changes the corpus, not local learning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from balanced_interruption_exposure import make_balanced_cases


def test_balanced_exposure_adds_all_generic_training_combinations():
    original, added = make_balanced_cases()
    assert len(original) == 128
    assert len(added) == 64
    assert sum(family == 'interruption' for family, _, _ in original) == 16
    assert {family for family, _, _ in added} == {'interruption'}
    assert len({shape for _, shape, _ in added}) == 4
