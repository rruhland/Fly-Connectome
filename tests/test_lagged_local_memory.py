"""Causal local hidden memory preserves recent sites for delayed credit."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from lagged_local_memory import LagConditionalForecast, RecentLatentMemory


def impulse():
    state = torch.zeros((1, 32, 64))
    state[0, 12, 20] = 1
    return state


def target():
    event = torch.zeros((2, 32, 64))
    event[1, 13, 20] = 1
    return event


def test_recent_site_survives_quiet_frames_and_expires_at_bound():
    memory = RecentLatentMemory(1, window=4)
    memory.step(impulse())
    for elapsed in range(1, 4):
        memory.step(torch.zeros_like(impulse()))
        assert memory.activity[0, 12, 20] == 1
        assert memory.age[0, 12, 20] == elapsed
    memory.step(torch.zeros_like(impulse()))
    assert memory.activity[0, 12, 20] == 0


def test_lagged_spatial_credit_updates_only_arriving_age_channel():
    memory = RecentLatentMemory(1, window=4)
    memory.step(impulse())
    memory.step(torch.zeros_like(impulse()))
    head = LagConditionalForecast(1, window=4, horizon=1, eta=1)
    head.step(memory.activity, memory.age, torch.zeros_like(target()),
              learn=True)
    head.step(torch.zeros_like(impulse()), memory.age, target(), learn=True)
    assert head.weights[0, 1, 1, 3, 2] == 1
    assert head.weights[0, 0].sum() == 0
    head.reset_state()
    assert head.step(memory.activity, memory.age, target())[1, 13, 20] == 1


def test_four_frame_target_cannot_update_before_due():
    memory = RecentLatentMemory(1, window=8)
    memory.step(impulse())
    head = LagConditionalForecast(1, window=8, horizon=4, eta=1)
    head.step(memory.activity, memory.age, target(), learn=True)
    for _ in range(3):
        head.step(torch.zeros_like(impulse()), memory.age,
                  torch.zeros_like(target()), learn=True)
    assert head.weights.sum() == 0
    head.step(torch.zeros_like(impulse()), memory.age, target(), learn=True)
    assert head.weights[0, 0, 1, 3, 2] == 1
