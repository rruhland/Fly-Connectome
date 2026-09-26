"""The opt-in temporal state uses local, separate transition and emission credit."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from recurrent_temporal_state import RecurrentTemporalState


def test_transition_credit_changes_only_near_active_source():
    model = RecurrentTemporalState(height=8, width=10)
    event = torch.zeros((2, 8, 10))
    event[1, 4, 4] = 1.
    source = model.step(event)
    before = model.transition.clone()
    target = torch.zeros_like(event)
    target[1, 4, 5] = 1.
    model.credit_transition(source, target)
    assert not torch.equal(model.transition, before)
    assert model.emission.abs().sum() == 0


def test_emission_credit_is_independent_and_quiet_target_reduces_emission():
    model = RecurrentTemporalState(height=8, width=10)
    event = torch.zeros((2, 8, 10))
    event[1, 4, 4] = 1.
    source = model.step(event)
    target = torch.zeros_like(event)
    target[1, 4, 5] = 1.
    transition = model.transition.clone()
    model.credit_emission(source, target)
    learned = model.forecast(source)[1, 4, 5]
    assert learned > 0
    model.credit_emission(source, torch.zeros_like(target))
    assert model.forecast(source)[1, 4, 5] < learned
    assert torch.equal(model.transition, transition)


def test_recurrence_can_affect_quiet_state_and_reset_removes_history():
    model = RecurrentTemporalState(height=8, width=10)
    event = torch.zeros((2, 8, 10))
    event[1, 4, 4] = 1.
    model.step(event)
    with_history = model.step(torch.zeros_like(event)).sum()
    assert with_history > 0
    model.reset_state()
    assert model.step(torch.zeros_like(event)).sum() == 0
