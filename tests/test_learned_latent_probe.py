"""Behavioral checks for the opt-in unlabeled latent-learning probe."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from learned_latent_probe import LocalVisualLatent


def test_local_recurrent_credit_arrives_only_after_next_observation():
    model = LocalVisualLatent(channels=4, seed=0)
    first = torch.zeros((2, 32, 64))
    second = torch.zeros_like(first)
    first[0, 16, 16] = 1
    second[0, 16, 17] = 1
    initial = model.recurrent.clone()
    model.step(first, learn=True)
    assert torch.equal(model.recurrent, initial)
    model.step(second, learn=True)
    assert not torch.equal(model.recurrent, initial)
    assert not any(parameter.requires_grad for parameter in
                   (model.sensory, model.recurrent))


def test_sensory_dictionary_changes_from_unlabeled_local_events():
    model = LocalVisualLatent(channels=4, seed=1)
    initial = model.sensory.clone()
    event = torch.zeros((2, 32, 64))
    event[1, 10:13, 20] = 1
    model.step(event, learn=True)
    assert not torch.equal(model.sensory, initial)
