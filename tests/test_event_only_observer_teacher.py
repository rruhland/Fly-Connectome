"""Observer teaching evidence must be available from the event stream."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from event_only_observer_teacher import local_event_support


def test_isolated_event_lacks_local_support():
    raw = torch.zeros((2, 8, 8))
    raw[0, 4, 4] = 1.
    assert local_event_support(raw, torch.zeros_like(raw))[0, 4, 4] == 0


def test_nearby_past_or_simultaneous_event_provides_support():
    raw = torch.zeros((2, 8, 8))
    raw[0, 4, 4] = 1.
    previous = torch.zeros_like(raw)
    previous[0, 4, 3] = .5
    assert local_event_support(raw, previous)[0, 4, 4] == 1
    raw[0, 4, 5] = 1.
    assert bool(local_event_support(raw, torch.zeros_like(raw))
                [0, 4, 4:6].all())
