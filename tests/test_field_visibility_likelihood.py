"""Coarse local evidence credit can read subthreshold hidden activity."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from field_visibility_likelihood import FieldVisibilityHead
from learned_transition_units import TransitionPopulation
from local_hidden_transition import LocalHiddenTransition
from occlusion_identifiability import matched_prefix_suite
from run_field_visibility_likelihood import evaluate


class SubthresholdState:
    def __init__(self):
        self.pending = torch.zeros((2, 32, 64))
        self.pending[0, 16, 16] = .2
        self.pending_age = torch.ones_like(self.pending)
        self.observed = torch.zeros_like(self.pending)


def test_field_head_uses_subthreshold_hidden_state_and_delayed_event():
    state = SubthresholdState()
    head = FieldVisibilityHead(eta=1)
    blank = torch.zeros((2, 32, 64))
    first = head.step(state, blank).clone()
    assert first.shape == (3, 7)
    assert first.max() == .5
    target = blank.clone()
    target[0, 16, 17] = 1
    head.step(state, target, learn=True)
    assert head.probability.max() > first.max()


def test_no_hidden_mass_has_zero_field_probability():
    state = SubthresholdState()
    state.pending.zero_()
    head = FieldVisibilityHead()
    assert head.step(state, torch.zeros((2, 32, 64))).sum() == 0


def test_field_evaluation_keeps_matched_prefix_probabilities_equal():
    transition = LocalHiddenTransition(TransitionPopulation(channels=16))
    heads = {name: FieldVisibilityHead()
             for name in ('learned', 'shuffled_time')}
    result = evaluate(transition, heads, .5, [],
                      matched_prefix_suite()[:1])
    assert result['identical_prefix_probabilities'] == {
        'two_vs_three': 1, 'three_vs_four': 1}
