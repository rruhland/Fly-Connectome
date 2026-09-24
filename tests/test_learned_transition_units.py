"""Behavioral checks for the opt-in learned local transition population."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from learned_transition_units import TransitionPopulation


def test_current_code_assignment_depends_on_local_previous_code():
    model = TransitionPopulation(channels=2, units=2, seed=0,
                                 homeostasis=False)
    model.dictionary.zero_()
    model.dictionary[0, 1*25+12] = 1
    model.dictionary[1, 1*25+12] = 1
    model.dictionary[1, (2+0)*25+11] = 1
    previous = torch.zeros((2, 32, 64))
    current = torch.zeros_like(previous)
    previous[0, 16, 15] = 1
    current[1, 16, 16] = 1
    model.step(previous, learn_dictionary=False)
    model.step(current, learn_dictionary=False)
    assert model.latent[1, 16, 16] == 1
    model.reset_state()
    model.step(current, learn_dictionary=False)
    assert model.latent[0, 16, 16] == 1


def test_predictive_synapse_updates_after_target_observation():
    model = TransitionPopulation(channels=2, units=2, seed=0,
                                 homeostasis=False)
    model.dictionary.zero_()
    model.dictionary[0, 0*25+12] = 1
    first = torch.zeros((2, 32, 64))
    second = torch.zeros_like(first)
    first[0, 16, 15] = 1
    second[0, 16, 16] = 1
    initial = model.predictive.clone()
    model.step(first, learn_dictionary=False, learn_prediction=True)
    assert torch.equal(model.predictive, initial)
    model.step(second, learn_dictionary=False, learn_prediction=True)
    assert not torch.equal(model.predictive, initial)


def test_shuffled_credit_target_does_not_change_inferred_code():
    ordered = TransitionPopulation(channels=2, units=2, seed=0,
                                   homeostasis=False)
    shuffled = TransitionPopulation(channels=2, units=2, seed=0,
                                    homeostasis=False)
    for model in (ordered, shuffled):
        model.dictionary.zero_()
        model.dictionary[0, 12] = 1
    first = torch.zeros((2, 32, 64))
    second = torch.zeros_like(first)
    first[0, 16, 15] = 1
    second[0, 16, 16] = 1
    ordered.step(first, learn_prediction=True)
    shuffled.step(first, learn_prediction=True)
    ordered.step(second, learn_prediction=True)
    shuffled.step(second, learn_prediction=True,
                  credit_target=torch.zeros_like(second))
    assert torch.equal(ordered.latent, shuffled.latent)
    assert ordered.predictive.sum() > shuffled.predictive.sum()
