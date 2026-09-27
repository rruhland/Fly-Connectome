"""Reconstruction learns only from declared observations and remains causal."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from joint_reconstruction import TemporalPatchObserver, SparseFrameObserver


def test_temporal_dictionary_freeze_and_reset_preserve_learned_weights():
    model = TemporalPatchObserver(height=8, width=8)
    for x in range(1, 6):
        raw = torch.zeros(2, 8, 8)
        raw[1, 3, x] = 1.
        model.step(raw, learn=True)
    saved = model.dictionary.clone()
    assert saved.abs().sum() > 0
    model.reset_state()
    assert torch.equal(model.dictionary, saved)
    model.step(torch.zeros(2, 8, 8))
    assert torch.equal(model.dictionary, saved)
    assert model.contrast.abs().sum() == 0


def test_sparse_teacher_absence_does_not_train_and_current_frame_does():
    model = SparseFrameObserver(height=8, width=8)
    raw = torch.zeros(2, 8, 8)
    raw[1, 3, 3] = 1.
    saved = model.weights.clone()
    model.step(raw, learn=True)
    assert torch.equal(model.weights, saved)
    absolute = torch.zeros(8, 8)
    absolute[3, 4] = 1.
    model.step(torch.zeros_like(raw), absolute=absolute, learn=True)
    assert not torch.equal(model.weights, saved)
    assert torch.equal(model.contrast, absolute)


def test_sparse_frame_learning_can_be_frozen_and_reset_clears_history():
    model = SparseFrameObserver(height=8, width=8)
    model.step(torch.ones(2, 8, 8), absolute=torch.ones(8, 8), learn=True)
    saved = model.weights.clone()
    model.step(torch.zeros(2, 8, 8), absolute=torch.zeros(8, 8), learn=False)
    assert torch.equal(model.weights, saved)
    model.reset_state()
    assert model.contrast.abs().sum() == 0
