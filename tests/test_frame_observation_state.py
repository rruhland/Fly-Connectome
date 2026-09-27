"""Full image evidence is observed even when the event stream is quiet."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from frame_observation_state import FrameObservationState


def sample(x, *, event=True):
    value = torch.zeros(12, 16, 32)
    value[6, 8, x] = 1.
    if event:
        value[1, 8, x] = 1.
    return value


def test_first_movement_does_not_assume_known_zero_velocity():
    model = FrameObservationState(height=16, width=32)
    for x in (3, 7, 11):
        model.step(sample(x))
    assert len(model.entities) == 1
    assert model.entities[0]['id'] == 0
    assert torch.equal(model.entities[0]['position'], torch.tensor([8., 11.]))


def test_stopping_remains_observed_and_retains_identity():
    model = FrameObservationState(height=16, width=32)
    for x in (3, 4, 5):
        model.step(sample(x))
    identity = model.entities[0]['id']
    for _ in range(6):
        model.step(sample(5, event=False))
    entity = next(e for e in model.entities if e['id'] == identity)
    assert torch.equal(entity['position'], torch.tensor([8., 5.]))
    assert entity['strength'] == 1.
    model.step(sample(6))
    assert next(e for e in model.entities if e['id'] == identity)['position'][1] == 6


def test_outage_cannot_reconfirm_stale_image():
    model = FrameObservationState(height=16, width=32)
    for x in (3, 4, 5):
        model.step(sample(x))
    model.step(sample(5, event=False), observation_available=False)
    assert model.tracker.slots[0]['last_seen'] == 2
    assert model.entities[0]['position'][1] == 6
