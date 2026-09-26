"""Causal corroboration suppresses isolated noise before tracklet birth."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from corroborated_context_memory import CorroboratedContextMemory


def sample(x, strength):
    value = torch.zeros((12, 8, 16))
    value[0, 4, x] = strength
    return value


def test_isolated_weak_event_does_not_create_tracklet():
    model = CorroboratedContextMemory(height=8, width=16)
    model.step(sample(4, .2))
    assert model.tracker.slots == []


def test_neighboring_recent_event_recovers_full_strength():
    model = CorroboratedContextMemory(height=8, width=16)
    model.step(sample(4, .8))
    model.step(sample(5, .8))
    assert len(model.tracker.slots) == 1
    assert float(model.tracker.slots[0]['velocity'][1]) > 0
