"""Visibility credit is delayed, local, and separate from hidden motion."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_visibility_likelihood import LocalVisibilityHead
from run_local_visibility_likelihood import training_cases


class PredictedState:
    def __init__(self):
        self.pending = torch.zeros((2, 32, 64))
        self.pending[0, 16, 16] = 1
        self.pending_age = torch.ones_like(self.pending)
        self.observed = torch.zeros_like(self.pending)


def test_next_visible_event_updates_only_local_probability_head():
    state = PredictedState()
    head = LocalVisibilityHead(eta=1)
    blank = torch.zeros((2, 32, 64))
    head.step(state, blank, learn=True)
    first = float(head.probability[16, 16])
    target = blank.clone()
    target[0, 16, 17] = 1
    head.step(state, target, learn=True)
    assert head.probability[16, 16] > first
    assert state.pending[0, 16, 16] == 1


def test_no_hidden_candidate_emits_no_visibility_probability():
    state = PredictedState()
    state.pending.zero_()
    head = LocalVisibilityHead()
    head.step(state, torch.zeros((2, 32, 64)))
    assert head.probability.sum() == 0


def test_visibility_training_uses_clean_and_variable_gap_examples():
    cases = training_cases()
    assert len(cases) == 480
    assert sum(kind == 'clean' for kind, _ in cases) == 96
