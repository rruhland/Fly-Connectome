"""A frozen gain selects events using forecast mass available at issue time."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from event_budget_probe import select_by_mass


def test_budget_selects_highest_local_scores_without_target_information():
    forecast = torch.zeros((2, 32, 64))
    forecast[0, 8, 9] = .7
    forecast[1, 9, 9] = .4
    selected = select_by_mass(forecast, gain=1)
    assert selected.sum() == 1
    assert selected[0, 8, 9]
    assert not selected[1, 9, 9]


def test_zero_mass_emits_nothing():
    assert select_by_mass(torch.zeros((2, 32, 64)), gain=10).sum() == 0
