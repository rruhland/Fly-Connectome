"""Shared representation probes keep labels outside model learning."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from representation_family_audit import (SensoryControl, capture,
                                         fit_occupancy_probe, score_cases)
from run_decisive_representation_audit import make_cases, object_mask


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


def test_frozen_probes_recognize_a_known_spatial_code():
    cases = [case for case in make_cases() if case['split'] == 'calibration']
    directions = ('up', 'down', 'left', 'right')
    episodes = []
    for case in cases:
        direction = directions.index(case['objects'][0]['direction'])
        states = []
        for frame in range(18):
            state = torch.zeros((6, 32, 64))
            state[direction, object_mask(case['objects'][0], frame)] = 1.
            states.append(state)
        episodes.append(states)
    scores = score_cases(SensoryControl(), cases, episodes)
    assert scores['splits']['calibration']['occupancy_f1'] == 1.
    assert scores['splits']['calibration']['direction_accuracy'] == 1.
