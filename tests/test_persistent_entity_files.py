"""Online entity hypotheses arise from visual evidence without true IDs."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from persistent_entity_files import PersistentEntityFiles


def event(*, on=(), off=()):
    result = torch.zeros((2, 12, 20))
    for y, x in on:
        result[1, y, x] = 1.
    for y, x in off:
        result[0, y, x] = 1.
    return result


def test_supported_shape_forms_one_hypothesis_and_survives_quiet():
    model = PersistentEntityFiles(height=12, width=20)
    model.step(event(on=((5, 5), (5, 6), (5, 7))))
    assert len(model.slots) == 1
    identity = model.slots[0]['id']
    for _ in range(4):
        model.step(event())
    assert len(model.slots) == 1
    assert model.slots[0]['id'] == identity


def test_unsupported_single_event_does_not_become_entity():
    model = PersistentEntityFiles(height=12, width=20)
    model.step(event(on=((5, 5),)))
    assert not model.slots


def test_local_prediction_error_updates_motion_and_next_event_forecast():
    model = PersistentEntityFiles(height=12, width=20)
    model.step(event(on=((5, 5), (5, 6))))
    model.step(event(on=((5, 7),), off=((5, 5),)))
    assert len(model.slots) == 1
    assert model.slots[0]['velocity'][1] > 0
    forecast = model.forecast()
    assert forecast[1, 5, 8] > 0
    assert forecast[0, 5, 6] > 0


def test_conflicting_merged_evidence_keeps_two_hypotheses():
    model = PersistentEntityFiles(height=12, width=20)
    model.step(event(on=((5, 4), (5, 5), (5, 14), (5, 15))))
    assert len(model.slots) == 2
    model.step(event(on=((5, 6), (5, 13)), off=((5, 4), (5, 15))))
    initial_ids = {slot['id'] for slot in model.slots}
    model.step(event(on=((5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12)),
                     off=((5, 5), (5, 6), (5, 13), (5, 14))))
    assert initial_ids.issubset({slot['id'] for slot in model.slots})
