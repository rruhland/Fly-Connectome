"""Behavioral check for the opt-in empirical local transition diagnostic."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from empirical_transition_readout import conditional_weights
from learned_transition_units import TransitionPopulation


def test_local_conditional_weight_counts_observed_future_at_offset():
    model = TransitionPopulation(channels=2, units=2, seed=0,
                                 homeostasis=False)
    model.dictionary.zero_()
    model.dictionary[0, 12] = 1
    first = torch.zeros((2, 32, 64))
    second = torch.zeros_like(first)
    first[0, 16, 15] = 1
    second[0, 16, 16] = 1
    weights = conditional_weights(model, [[first, second]])
    assert weights[0, 0, 2, 3] == 1
    assert weights.sum() == 1
