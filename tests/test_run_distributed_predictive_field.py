"""Distributed-field controls receive identical next-event targets."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from distributed_predictive_field import DistributedPredictiveField
from run_distributed_predictive_field import evaluate_streams


def event(*, on=(), off=()):
    result = torch.zeros((2, 8, 10))
    for y, x in on:
        result[1, y, x] = 1.
    for y, x in off:
        result[0, y, x] = 1.
    return result


def test_all_arms_score_same_future_clean_camera_event():
    first = event(on=((4, 4),))
    second = event(on=((4, 5),), off=((4, 4),))
    unrelated = [event(on=((5, 8),)), event(off=((5, 8),))]
    trained = {name: DistributedPredictiveField(height=8, width=10)
               for name in ('aligned', 'shuffled_credit')}
    scores, state = evaluate_streams(
        [([first, second], [first, second], unrelated)], trained)
    assert all(row['events'] == 1 and row['targets'] == 2
               for row in scores.values())
    assert all(row['active_frames'] == 2 for row in state.values())
