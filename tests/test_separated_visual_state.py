"""Local recurrent state and visible-event readout stay separate."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from learned_transition_units import TransitionPopulation
from separated_visual_state import SeparatedVisualState


def test_blank_input_can_keep_latent_without_emitting_event():
    encoder = TransitionPopulation(channels=16, seed=0)
    model = SeparatedVisualState(encoder, recurrence=True)
    model.state_weights[:, :, 2, 3] = 1
    current = torch.zeros((16, 32, 64))
    current[1, 16, 16] = 1
    blank_events = torch.zeros((2, 32, 64))
    model.step(current, blank_events)
    predicted_events = model.step(torch.zeros_like(current), blank_events)
    assert model.observed.sum() == 0
    assert model.imagined.sum() > 0
    assert predicted_events.sum() == 0


def test_delayed_visible_event_credit_changes_only_local_emission_weight():
    encoder = TransitionPopulation(channels=16, seed=0)
    model = SeparatedVisualState(encoder, recurrence=False,
                                 emission_eta=1.)
    current = torch.zeros((16, 32, 64))
    current[1, 16, 16] = 1
    blank_events = torch.zeros((2, 32, 64))
    target = blank_events.clone()
    target[0, 16, 17] = 1
    model.step(current, blank_events, learn=True)
    model.step(torch.zeros_like(current), target, learn=True)
    assert model.emission_observed[0, :, 2, 3].sum() > 0
    assert model.emission_observed[1].sum() == 0
    assert model.emission_imagined.sum() == 0
