"""Event-indexed execution skips empty local-code frames."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_event_clock_transfer import advance


class CountingModel:
    def __init__(self):
        self.advances = []

    def step(self, code, *, advance=True):
        self.advances.append(advance)


def test_event_clock_skips_empty_code_but_frame_clock_advances():
    model = CountingModel()
    blank = torch.zeros((16, 32, 64))
    impulse = blank.clone()
    impulse[0, 16, 16] = 1
    advance(model, blank, clock='event')
    advance(model, impulse, clock='event')
    advance(model, blank, clock='frame')
    assert model.advances == [False, True, True]
