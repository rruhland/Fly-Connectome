"""Shared representation probes keep labels outside model learning."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from representation_family_audit import SensoryControl, capture, fit_occupancy_probe


def test_sensory_control_returns_only_supplied_planes():
    inputs = [torch.rand((6, 8, 10)), torch.zeros((6, 8, 10))]
    states = capture(SensoryControl(), inputs)
    assert len(states) == 2
    assert torch.equal(states[0], inputs[0])
    assert torch.equal(states[1], inputs[1])


def test_local_occupancy_probe_uses_calibration_labels_only():
    positive = torch.tensor([[[1., 0.], [1., 0.]]])
    mask = torch.tensor([[True, False], [True, False]])
    weight, threshold = fit_occupancy_probe([(positive, mask)])
    assert bool(((weight[:, None, None]*positive).sum(0) > threshold)[0, 0])
    assert not bool(((weight[:, None, None]*positive).sum(0) > threshold)[0, 1])
