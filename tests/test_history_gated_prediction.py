"""The local event readout credits only previously active history sources."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from history_gated_code import HistoryGatedCode
from history_gated_prediction import (LocalEventReadout,
                                      expanded_interruption_cases)
from learned_transition_units import TransitionPopulation


def test_first_sighting_without_history_cannot_update_event_weights():
    code = HistoryGatedCode(TransitionPopulation(channels=16, units=24))
    model = LocalEventReadout(code)
    events = torch.zeros((2, 32, 64))
    events[0, 16, 20] = 1
    coincidence = torch.zeros((16, 32, 64))
    assert model.step(events, coincidence, learn=True).sum() == 0
    assert model.step(events, coincidence, learn=True).sum() == 0
    assert model.weights.sum() == 0


def test_expanded_interruption_suite_covers_all_directions_and_positions():
    cases = expanded_interruption_cases()
    assert len(cases) == 144
    assert {family for family, _, _ in cases} == {
        'occlusion:up', 'occlusion:down',
        'occlusion:left', 'occlusion:right'}
