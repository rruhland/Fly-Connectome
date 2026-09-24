"""Causal and polarity checks for low-level correlation input."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from correlation_input_latent import correlation_sequence, primitive_to_events


def test_correlation_primitive_is_local_causal_and_signed():
    previous = torch.zeros((2, 32, 64))
    current = torch.zeros_like(previous)
    previous[0, 16, 15] = 1
    current[0, 16, 16] = 1
    code = correlation_sequence([previous, current])
    assert len(code) == 2
    assert code[0].sum() == 0
    assert code[1].shape == (16, 32, 64)
    assert code[1][1, 16, 16] == 1
    assert code[1].sum() == 1
    mismatched = current.flip(0)
    assert correlation_sequence([previous, mismatched])[1].sum() == 0


def test_primitive_prediction_decodes_polarity_without_direction_label():
    predicted = torch.zeros((16, 32, 64))
    predicted[0, 16, 16] = .3
    predicted[1, 16, 16] = .4
    predicted[8, 12, 15] = .6
    events = primitive_to_events(predicted)
    assert torch.isclose(events[0, 16, 16], torch.tensor(.7))
    assert torch.isclose(events[1, 12, 15], torch.tensor(.6))
    assert torch.isclose(events.sum(), torch.tensor(1.3))
