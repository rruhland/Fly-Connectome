"""Behavioral checks for the opt-in generic visual grouping baseline."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from generic_entity_baseline import EventOccupancy, components


def test_event_occupancy_persists_through_quiet_and_clears_on_reversal():
    memory = EventOccupancy()
    entering = torch.zeros((2, 32, 64))
    entering[0, 10, 20] = 1
    assert memory.step(entering)[10, 20]
    assert memory.step(torch.zeros_like(entering))[10, 20]
    leaving = torch.zeros_like(entering)
    leaving[1, 10, 20] = 1
    assert not memory.step(leaving)[10, 20]


def test_components_keep_separated_visual_regions_distinct():
    mask = torch.zeros((32, 64), dtype=torch.bool)
    mask[10:13, 20:23] = True
    mask[10:13, 30:33] = True
    found = components(mask)
    assert len(found) == 2
    assert [round(group[1]) for group in found] == [21, 31]
