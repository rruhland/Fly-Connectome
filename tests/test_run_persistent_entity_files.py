"""Entity forecast arms receive the same next observed event target."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_persistent_entity_files import evaluate_forecasts
from sparse_recurrent_field import SparseRecurrentField


def test_matched_full_frame_forecast_accounting():
    first = torch.zeros((2, 32, 64))
    second = first.clone()
    first[1, 10, 10:13] = 1.
    second[1, 10, 13] = 1.
    second[0, 10, 10] = 1.
    fast = SparseRecurrentField(units=1)
    rows = evaluate_forecasts(fast, [[first, second]])
    assert all(row['events'] == 1 and row['targets'] == 2
               for row in rows.values())
