"""Local selector compares temporal pathway errors without backpropagation."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_temporal_mixture import LocalTemporalMixture


def test_equal_untrained_mixture_and_local_error_selects_better_pathway():
    model = LocalTemporalMixture(height=8, width=10)
    slow_state = torch.zeros((8, 8, 10))
    fast_state = torch.zeros_like(slow_state)
    slow_state[1, 4, 4] = 1.
    fast_state[1, 4, 4] = 1.
    slow = torch.zeros((2, 8, 10))
    fast = torch.zeros_like(slow)
    target = torch.zeros_like(slow)
    slow[1, 4, 5] = 1.
    target[1, 4, 5] = 1.
    features = model.features(slow_state, fast_state)
    assert model.forecast(features, slow, fast)[1, 4, 5] == .5
    model.credit(features, slow, fast, target)
    assert model.forecast(features, slow, fast)[1, 4, 5] > .5
    assert model.weights[0].abs().sum() == 0


def test_selector_reads_only_bounded_local_visual_context():
    model = LocalTemporalMixture(height=8, width=10)
    slow_state = torch.zeros((8, 8, 10))
    slow_state[1, 4, 4] = 1.
    features = model.features(slow_state, torch.zeros_like(slow_state))
    assert features[:, 4, 5].sum() > 0
    assert features[:, 0, 0].sum() == 0
