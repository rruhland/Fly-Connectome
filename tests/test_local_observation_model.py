"""Local event credibility learns from independent signed visual change."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_observation_model import LocalObservationModel


def event(*, on=(), off=()):
    result = torch.zeros((2, 8, 10))
    for y, x in on:
        result[1, y, x] = 1.
    for y, x in off:
        result[0, y, x] = 1.
    return result


def test_observer_emits_continuous_evidence_only_where_event_arrives():
    model = LocalObservationModel(height=8, width=10)
    observed, features = model.step(event(on=((4, 4),)))
    assert observed.shape == (2, 8, 10)
    assert 0 < observed[1, 4, 4] < 1
    assert int((observed > 0).sum()) == 1
    assert features.shape[0] == 8


def test_local_paired_visual_credit_changes_event_credibility():
    true = LocalObservationModel(height=8, width=10)
    false = LocalObservationModel(height=8, width=10)
    raw = event(on=((4, 4),))
    for _ in range(4):
        _, true_features = true.step(raw)
        true.credit(true_features, raw, raw)
        true.reset_state()
        _, false_features = false.step(raw)
        false.credit(false_features, raw, event())
        false.reset_state()
    assert true.step(raw)[0][1, 4, 4] > false.step(raw)[0][1, 4, 4]


def test_absolute_visual_evidence_changes_memory_gradually_after_filtering():
    model = LocalObservationModel(height=8, width=10)
    raw = event(on=((4, 4),))
    filtered, _ = model.step(raw, absolute=torch.zeros((8, 10)))
    assert filtered[1, 4, 4] == .5
    assert 0 < model.contrast[4, 4] < .5
