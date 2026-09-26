"""Control scoring uses a frozen visible-only calibration."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from visual_history_scoring import fit_visible_readout, hidden_rank, pair_separation


def test_visible_readout_cannot_use_hidden_target():
    visible = torch.zeros((2, 4, 8))
    visible[0, 1, 2] = 1.
    mask = torch.zeros((4, 8), dtype=torch.bool)
    mask[1, 2] = True
    readout = fit_visible_readout([(visible, mask)])
    field = readout(visible)
    assert field[1, 2] > field[1, 3]


def test_hidden_rank_counts_true_site_in_top_k():
    field = torch.zeros((4, 8))
    field[2, 6] = 3.
    mask = torch.zeros_like(field, dtype=torch.bool)
    mask[2, 6] = True
    assert hidden_rank(field, mask, k=1) == 1.
    field[1, 1] = 4.
    assert hidden_rank(field, mask, k=1) == 0.


def test_pair_separation_uses_entire_state():
    left = torch.zeros((2, 4, 8))
    right = left.clone()
    right[1, 2, 6] = 1.
    assert pair_separation(left, right) == 1.
