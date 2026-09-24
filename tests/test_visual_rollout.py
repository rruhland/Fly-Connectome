"""Closed-loop diagnostic must restore the online state after imagination."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from visual_rollout import rollout_from_state
from learned_transition_units import TransitionPopulation


def test_rollout_restores_state_for_following_real_evidence():
    model = TransitionPopulation(channels=16, seed=0)
    model.predictive[1, :, 2, 3] = 1
    current = torch.zeros((16, 32, 64))
    current[1, 16, 16] = 1
    following = torch.zeros_like(current)
    following[1, 16, 17] = 1
    model.step(current)
    expected = model.step(following)
    model.reset_state()
    first = model.step(current)
    forecasts = rollout_from_state(model, first, steps=4)
    assert len(forecasts) == 4
    actual = model.step(following)
    assert torch.equal(actual, expected)
