"""Separate local event and false-alarm credit uses no distant target."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from balanced_trace_credit import update_balanced


def test_balanced_credit_confirms_event_and_suppresses_false_alarm():
    weights = torch.zeros((2, 24, 17, 17))
    weights[0, 2, 8, 7] = .5
    target = torch.zeros((2, 32, 64))
    target[0, 16, 33] = 1
    prediction = torch.zeros_like(target)
    prediction[0, 16, 31] = .5
    update_balanced(weights, [(2, 16, 32, 1.)], target,
                    prediction, .5)
    assert weights[0, 2, 8, 9] == .5
    assert weights[0, 2, 8, 7] == .25
    assert weights[1].sum() == 0
