"""Last observed ON/OFF evidence persists while untouched pixels stay unknown."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_observation_surface import LocalObservationSurface


def test_local_on_off_state_survives_quiet_frames():
    surface = LocalObservationSurface()
    event = torch.zeros((2, 32, 64))
    event[1, 16, 16] = 1
    surface.step(event)
    assert surface.known[16, 16]
    assert surface.value[16, 16] == 1
    assert not surface.known[16, 17]
    surface.step(torch.zeros_like(event))
    assert surface.value[16, 16] == 1
    off = torch.zeros_like(event)
    off[0, 16, 16] = 1
    surface.step(off)
    assert surface.value[16, 16] == 0
    assert surface.channels[0, 16, 16] == 1
    assert surface.channels[1, 16, 16] == 0


def test_reset_restores_unknown_everywhere():
    surface = LocalObservationSurface()
    event = torch.zeros((2, 32, 64))
    event[1, 8, 9] = 1
    surface.step(event)
    surface.reset_state()
    assert surface.known.sum() == 0
    assert surface.channels.sum() == 0
