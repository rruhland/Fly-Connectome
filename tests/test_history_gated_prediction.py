"""The local event readout credits only previously active history sources."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from history_gated_code import HistoryGatedCode
from history_gated_prediction import (LocalEventReadout, unit_local_update,
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


def test_unit_local_update_normalizes_each_units_own_sources():
    weights = torch.zeros((2, 3, 5, 5))
    error = torch.zeros((2, 32, 64))
    error[0, 10, 10] = 1
    error[0, 11, 10] = 1
    error[0, 10, 20] = 1
    sources = [(10, 10, 1, 1.), (11, 10, 1, 1.),
               (10, 20, 2, 1.)]
    unit_local_update(weights, sources, error, .5)
    assert weights[0, 1, 2, 2] == .5
    assert weights[0, 2, 2, 2] == .5
    assert weights[0, 0].sum() == 0
