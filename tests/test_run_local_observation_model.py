"""Observation-model arms share a future event and use causal evidence."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from distributed_predictive_field import DistributedPredictiveField
from local_observation_model import LocalObservationModel
from run_absolute_refresh import contrast_frames
from run_local_observation_model import evaluate_streams


def event(*, on=(), off=()):
    result = torch.zeros((2, 8, 10))
    for y, x in on:
        result[1, y, x] = 1.
    for y, x in off:
        result[0, y, x] = 1.
    return result


def test_all_arms_score_the_same_clean_next_event():
    first = event(on=((4, 4),))
    second = event(on=((4, 5),), off=((4, 4),))
    unrelated = [event(on=((5, 8),)), event(off=((5, 8),))]
    observers = {name: LocalObservationModel(height=8, width=10)
                 for name in ('aligned', 'shuffled_credit')}
    field = DistributedPredictiveField(height=8, width=10)
    sequence = [first, second]
    scores, state = evaluate_streams(
        [(sequence, sequence, unrelated,
          contrast_frames(sequence), contrast_frames(unrelated))],
        observers, field)
    assert all(row['events'] == 1 and row['targets'] == 2
               for row in scores.values())
    assert all(row['active_frames'] == 2 for row in state.values())
