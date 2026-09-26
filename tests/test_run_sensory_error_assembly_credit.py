"""Matched forecast evaluation for sensory-error assembly credit."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_sensory_error_assembly_credit import evaluate_forecasts
from slow_predictive_assemblies import SlowPredictiveAssemblies
from sparse_recurrent_field import SparseRecurrentField


def test_full_frame_evaluation_counts_the_same_future_event_for_each_arm():
    first = torch.zeros((2, 8, 8))
    second = first.clone()
    first[0, 4, 3] = 1.
    second[0, 4, 4] = 1.
    fast = SparseRecurrentField(units=1, height=8, width=8,
                                max_winners=1)
    models = {name: SlowPredictiveAssemblies(
        fast_units=1, units=1, height=8, width=8, max_winners=1)
        for name in ('sensory_error', 'winner_error', 'frozen',
                     'shuffled_sensory')}
    rows = evaluate_forecasts(fast, models, [[first, second]])
    assert all(row['events'] == 1 and row['targets'] == 1
               for row in rows.values())
    assert len(rows['sensory_error']['cases']) == 1
