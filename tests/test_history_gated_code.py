"""History-only units must be silent when no prior motion reaches onset."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from history_gated_code import HistoryGatedCode, unit_histogram
from learned_transition_units import TransitionPopulation


def test_raw_first_sighting_without_history_has_no_history_sources():
    model = HistoryGatedCode(TransitionPopulation(channels=16, units=24))
    events = torch.zeros((2, 32, 64))
    events[0, 16, 20] = 1
    coincidence = torch.zeros((16, 32, 64))
    assert model.step(events, coincidence) == []
    assert model.history.latent.sum() == 0


def test_history_histogram_keeps_unit_identity():
    histogram = unit_histogram([(10, 10, 1), (10, 11, 1),
                                (10, 12, 2)], 4)
    assert torch.allclose(histogram, torch.tensor([0., 2/3, 1/3, 0.]))
