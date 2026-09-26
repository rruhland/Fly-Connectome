"""Local recurrent visual field has graded state and local predictive credit."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from distributed_predictive_field import DistributedPredictiveField


def event(*, on=(), off=()):
    result = torch.zeros((2, 8, 10))
    for y, x in on:
        result[1, y, x] = 1.
    for y, x in off:
        result[0, y, x] = 1.
    return result


def test_event_traces_decay_and_absolute_evidence_nudges_contrast():
    model = DistributedPredictiveField(height=8, width=10)
    model.step(event(on=((4, 4),)))
    assert model.state[0:2].sum() > 0
    blank = torch.zeros((8, 10))
    model.step(event(), absolute=blank)
    assert 0 < model.contrast[4, 4] < 1
    assert 0 < model.state[0:2].sum() < 1


def test_local_credit_changes_only_weights_with_eligible_source():
    model = DistributedPredictiveField(height=8, width=10)
    source = model.step(event(on=((4, 4),))).clone()
    model.credit(source, event(on=((4, 5),)))
    assert model.weights[:, 0].abs().sum() == 0
    assert model.weights[:, 1].abs().sum() > 0
    assert model.weights[1, 1].sum() > 0


def test_learned_prediction_reenters_local_state_without_backprop():
    model = DistributedPredictiveField(height=8, width=10)
    source = model.step(event(on=((4, 4),))).clone()
    model.credit(source, event(on=((4, 5),)))
    forecast = model.forecast()
    assert forecast[1, 4, 5] > 0
    model.step(event())
    assert model.state[-2:].sum() > 0
