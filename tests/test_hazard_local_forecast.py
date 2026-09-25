"""Local event timing and local spatial outcome learn on separate traces."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from hazard_local_forecast import HazardLocalForecast


def state():
    value = torch.zeros((1, 32, 64))
    value[0, 12, 20] = 1
    return value


def age(value):
    result = torch.zeros_like(state())
    result[0, 12, 20] = value
    return result


def event(y=13):
    value = torch.zeros((2, 32, 64))
    value[1, y, 20] = 1
    return value


def test_due_event_updates_hazard_and_spatial_distribution_after_four_frames():
    head = HazardLocalForecast(1, horizon=4, eta=1)
    head.step(state(), age(0), event(), learn=True)
    for _ in range(3):
        head.step(torch.zeros_like(state()), age(0),
                  torch.zeros_like(event()), learn=True)
    assert head.rate.sum() == 0
    head.step(torch.zeros_like(state()), age(0), event(), learn=True)
    assert head.rate[0, 0] == 1
    assert head.spatial[0, 1, 3, 2] == 1
    head.reset_state()
    prediction = head.step(state(), age(0), event())
    assert prediction[1, 13, 20] == 1


def test_quiet_credit_changes_rate_but_preserves_spatial_outcome():
    head = HazardLocalForecast(1, horizon=1, eta=.5)
    head.rate[0, 0] = 1
    head.spatial[0, 1, 3, 2] = 1
    head.step(state(), age(0), event(), learn=True)
    head.step(torch.zeros_like(state()), age(0),
              torch.zeros_like(event()), learn=True)
    assert head.rate[0, 0] == .5
    assert head.spatial[0, 1, 3, 2] == 1


def test_age_context_selects_distinct_local_rate():
    head = HazardLocalForecast(1, horizon=1, eta=1)
    head.step(state(), age(0), event(), learn=True)
    head.step(state(), age(5), event(), learn=True)
    head.step(torch.zeros_like(state()), age(0),
              torch.zeros_like(event()), learn=True)
    assert head.rate[0, 0] == 1
    assert head.rate[0, 3] == 0
