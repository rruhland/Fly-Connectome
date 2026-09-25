"""A 5×5 local readout can only reach nearby future event pixels."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from forecast_reachability import reached_targets


def test_reachability_counts_signed_targets_within_local_radius():
    sources = torch.zeros((2, 32, 64))
    sources[0, 12, 20] = 1
    targets = torch.zeros((2, 32, 64))
    targets[0, 14, 22] = 1
    targets[1, 15, 20] = 1
    assert reached_targets(sources, targets) == (1, 2)
    assert reached_targets(sources, targets, radius=3) == (2, 2)


def test_no_source_cannot_reach_future_event():
    targets = torch.zeros((2, 32, 64))
    targets[1, 12, 20] = 1
    assert reached_targets(torch.zeros((3, 32, 64)), targets) == (0, 1)
