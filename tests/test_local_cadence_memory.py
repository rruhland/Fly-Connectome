"""Visible event timing alone can credit local motion transitions."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_cadence_memory import LocalCadenceMemory


def sample(x):
    value = torch.zeros((12, 8, 32))
    if x is not None:
        value[0, 4, x] = 1.
    return value


def test_learns_visible_cadence_without_hidden_targets():
    model = LocalCadenceMemory(height=8, width=32)
    positions = (2, 3, 5, 8, 9, 11, 14, 15, 17, 20)
    for x in positions:
        model.step(sample(x), learn=True)
    assert model.transitions[2, 4] > 0
    assert model.transitions[4, 6] > 0
    assert model.transitions[6, 2] > 0


def test_learned_cadence_changes_autonomous_hidden_state():
    learned = LocalCadenceMemory(height=8, width=32)
    frozen = LocalCadenceMemory(height=8, width=32)
    positions = (2, 3, 5, 8, 9, 11, 14, 15, 17, 20, 21)
    for x in positions:
        learned.step(sample(x), learn=True)
        frozen.step(sample(x))
    for _ in range(3):
        future = learned.step(sample(None))
        baseline = frozen.step(sample(None))
    assert int(future[4].argmax()) > int(baseline[4].argmax())
