"""Evaluation helpers for the opt-in recurrent-field comparison."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_sparse_recurrent_field import evaluate_forecasts, rank_hits
from sparse_recurrent_field import SparseRecurrentField


def test_rank_hits_counts_only_target_events_at_top_sites():
    forecast = torch.zeros((2, 8, 8))
    forecast[0, 3, 4] = .9
    forecast[1, 2, 2] = .8
    target = torch.zeros_like(forecast)
    target[0, 3, 4] = 1.
    target[1, 2, 2] = 1.
    target[0, 5, 5] = 1.
    assert rank_hits(forecast, target, 1) == 1
    assert rank_hits(forecast, target, 2) == 2
    assert rank_hits(torch.zeros_like(forecast), target, 2) == 0


def test_forecast_evaluation_keeps_each_episode_for_paired_controls():
    first = torch.zeros((2, 8, 8))
    second = first.clone()
    first[0, 4, 3] = 1.
    second[0, 4, 4] = 1.
    models = {name: SparseRecurrentField(units=1, height=8,
                                          width=8, max_winners=1)
              for name in ('aligned', 'frozen', 'shuffled')}
    rows = evaluate_forecasts(models, [[first, second]])
    assert rows['aligned']['events'] == 1
    assert rows['repeat_event']['cases'][0]['targets'] == 1
    assert len(rows['aligned']['cases']) == 1
