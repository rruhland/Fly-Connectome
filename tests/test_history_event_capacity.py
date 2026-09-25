"""Raw event sign can be kept separate without changing history-unit identity."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from history_event_capacity import readout_sources


def test_readout_sources_crosses_unit_with_locally_observed_polarity():
    current = torch.zeros((2, 32, 64))
    current[0, 10, 11] = 1
    current[1, 10, 12] = 1
    sources = [(10, 11, 3), (10, 12, 3)]
    assert readout_sources(sources, current, units=8,
                           polarity_split=False) == sources
    assert readout_sources(sources, current, units=8,
                           polarity_split=True) == [
                               (10, 11, 3), (10, 12, 11)]
