"""Soft visual hypotheses keep identity through uncertain local evidence."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from graded_visual_state import GradedVisualState


def event(*, on=(), off=()):
    result = torch.zeros((2, 12, 20))
    for y, x in on:
        result[1, y, x] = 1.
    for y, x in off:
        result[0, y, x] = 1.
    return result


def image(*pixels):
    result = torch.zeros((12, 20))
    for y, x in pixels:
        result[y, x] = 1.
    return result


def test_isolated_event_remains_provisional_and_absolute_blank_does_not_delete_it():
    model = GradedVisualState(height=12, width=20)
    model.step(event(on=((5, 5), (5, 6))))
    assert len(model.hypotheses) == 1
    hypothesis = model.hypotheses[0]
    assert not model.live_slots
    before = hypothesis['confidence']
    model.step(event(), absolute=image())
    assert hypothesis in model.hypotheses
    assert 0 < hypothesis['confidence'] < before
    assert 0 < hypothesis['support'][5, 5] <= 1


def test_aligned_intensity_confirms_quiet_region_and_retains_identity():
    model = GradedVisualState(height=12, width=20)
    observed = image((5, 5), (5, 6))
    model.step(event(on=((5, 5), (5, 6))), absolute=observed)
    assert len(model.live_slots) == 1
    identity = model.live_slots[0]['id']
    for frame in range(1, 9):
        model.step(event(), absolute=observed if frame == 8 else None)
    assert len(model.live_slots) == 1
    assert model.live_slots[0]['id'] == identity


def test_local_prediction_error_changes_support_gradually_on_motion():
    model = GradedVisualState(height=12, width=20)
    model.step(event(on=((5, 5), (5, 6))),
               absolute=image((5, 5), (5, 6)))
    initial = model.live_slots[0]
    identity = initial['id']
    model.step(event(on=((5, 7),), off=((5, 5),)))
    assert len(model.live_slots) == 1
    changed = model.live_slots[0]
    assert changed['id'] == identity
    assert 0 < changed['support'][5, 5] < 1
    assert 0 < changed['support'][5, 7] < 1
    assert changed['velocity'][1] > 0
    assert model.forecast().shape == (2, 12, 20)


def test_two_separated_regions_keep_two_competing_hypotheses():
    model = GradedVisualState(height=12, width=20)
    first = image((5, 4), (5, 5), (5, 14), (5, 15))
    model.step(event(on=((5, 4), (5, 5), (5, 14), (5, 15))),
               absolute=first)
    identities = {row['id'] for row in model.live_slots}
    assert len(identities) == 2
    model.step(event(on=((5, 6), (5, 13)), off=((5, 4), (5, 15))))
    assert {row['id'] for row in model.live_slots} == identities


def test_shuffled_absolute_location_cannot_confirm_event_hypothesis():
    model = GradedVisualState(height=12, width=20)
    model.step(event(on=((5, 5), (5, 6))))
    original = model.hypotheses[0]
    model.step(event(), absolute=image((5, 15), (5, 16)))
    assert original['confidence'] < .5
    assert any(row['center'][1] > 10 for row in model.live_slots)
