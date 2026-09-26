"""Two visual predictors are compared at one fixed event budget."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_complementary_prediction import (collect_samples,
                                          reciprocal_rank_fusion,
                                          score_samples)
from sparse_recurrent_field import SparseRecurrentField


def test_reciprocal_rank_fusion_retains_two_distinct_strong_predictions():
    fast = torch.tensor([[[8., 2., 0., 0.]]])
    files = torch.tensor([[[0., 0., 2., 8.]]])
    winners = reciprocal_rank_fusion(fast, files).flatten().topk(2).indices
    assert set(winners.tolist()) == {0, 3}


def test_each_arm_uses_the_same_next_event_and_rank_budget():
    first = torch.zeros((2, 32, 64))
    second = first.clone()
    first[1, 5, 5:8] = 1.
    second[1, 5, 8] = 1.
    second[0, 5, 5] = 1.
    fast = SparseRecurrentField(units=1)
    samples = collect_samples(fast, [[first, second]], {})
    rows = score_samples(samples)
    assert all(row['events'] == 1 and row['targets'] == 2
               for row in rows.values())
