"""Matched full-frame scoring of paired slow assemblies."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from predictive_pair_assemblies import PredictivePairAssemblies
from run_predictive_pair_assemblies import evaluate_forecasts
from sparse_recurrent_field import SparseRecurrentField


def test_full_frame_evaluation_uses_same_target_for_all_arms():
    first = torch.zeros((2, 8, 8))
    second = first.clone()
    first[0, 4, 3] = 1.
    second[0, 4, 4] = 1.
    fast = SparseRecurrentField(units=1, height=8, width=8,
                                max_winners=1)
    models = {name: PredictivePairAssemblies(
        fast_units=1, units=1, height=8, width=8, max_winners=1)
        for name in ('aligned_pair', 'input_only', 'shuffled_pair')}
    rows = evaluate_forecasts(fast, models, [[first, second]])
    assert all(row['events'] == 1 and row['targets'] == 1
               for row in rows.values())
    assert all(len(row['cases']) == 1 for row in rows.values())
