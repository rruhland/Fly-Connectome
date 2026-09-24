"""Frozen predictive feedback must stay local and be resettable."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from learned_transition_units import TransitionPopulation
from predictive_feedback import FeedbackInference


def test_feedback_can_sustain_local_state_then_reset():
    model = TransitionPopulation(channels=16, seed=0)
    model.predictive[1, :, 2, 3] = 1
    wrapper = FeedbackInference(model, gain=.5)
    current = torch.zeros((16, 32, 64))
    current[1, 16, 16] = 1
    wrapper.step(current)
    wrapper.step(torch.zeros_like(current))
    assert model.latent.sum() > 0
    wrapper.reset_state()
    wrapper.step(torch.zeros_like(current))
    assert model.latent.sum() == 0
