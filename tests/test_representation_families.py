"""Each proposed family has its own local, gradient-free learning path."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from representation_families import FAMILIES


@pytest.mark.parametrize('name', FAMILIES)
def test_family_updates_only_during_local_learning(name):
    model = FAMILIES[name](height=8, width=12)
    initial = [tensor.clone() for tensor in model.learned_tensors()]
    inputs = []
    for frame in range(8):
        sample = torch.zeros((6, 8, 12))
        sample[frame % 2, 3, 3+frame % 4] = 1.
        sample[frame % 2, 3:5, 3:5] = 1.
        sample[2, 3, 3+frame % 4] = 1.
        sample[2, 3:5, 3:5] = 1.
        sample[4, 3, 3+frame % 4] = (-1.)**frame
        inputs.append(sample)
    for sample in inputs:
        state = model.step(sample)
        assert state.ndim == 3 and state.shape[1:] == (8, 12)
        assert torch.isfinite(state).all()
    assert all(torch.equal(before, after) for before, after in
               zip(initial, model.learned_tensors()))
    model.reset_state()
    for sample in inputs:
        model.step(sample, learn=True)
    assert any(not torch.equal(before, after) for before, after in
               zip(initial, model.learned_tensors()))
    assert all(not tensor.requires_grad for tensor in model.learned_tensors())
