"""Provisional visual files require local confirmation, and borrow local futures."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from predictive_file_evidence import PredictiveEvidenceFiles


def event(*, on=(), off=()):
    result = torch.zeros((2, 12, 20))
    for y, x in on:
        result[1, y, x] = 1.
    for y, x in off:
        result[0, y, x] = 1.
    return result


def test_unconfirmed_visual_region_is_provisional_then_expires():
    model = PredictiveEvidenceFiles(height=12, width=20)
    model.step(event(on=((5, 5), (5, 6))))
    assert len(model.slots) == 1
    assert not model.live_slots
    for _ in range(21):
        model.step(event())
    assert not model.slots


def test_reobserved_local_continuation_promotes_provisional_file():
    model = PredictiveEvidenceFiles(height=12, width=20)
    model.step(event(on=((5, 5), (5, 6))))
    model.step(event(on=((5, 7),), off=((5, 5),)))
    assert len(model.live_slots) == 1
    assert model.live_slots[0]['velocity'][1] > 0


def test_independent_signed_fast_support_can_confirm_birth():
    model = PredictiveEvidenceFiles(height=12, width=20)
    support = event(on=((5, 5),))
    model.step(event(on=((5, 5), (5, 6))), support=support)
    assert len(model.live_slots) == 1


def test_borrowed_fast_prediction_stays_near_confirmed_file():
    model = PredictiveEvidenceFiles(height=12, width=20,
                                    confirm_birth=False)
    model.step(event(on=((5, 5), (5, 6))))
    fast = event(on=((5, 8), (0, 0)))
    forecast = model.forecast(fast)
    assert forecast[1, 5, 8] > 0
    assert forecast[1, 0, 0] == 0
