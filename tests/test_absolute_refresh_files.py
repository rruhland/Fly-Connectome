"""Independent absolute contrast corrects event-only false occupancy."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from absolute_refresh_files import AbsoluteRefreshFiles


def event():
    result = torch.zeros((2, 12, 20))
    result[1, 5, 5:7] = 1.
    return result


def test_refresh_removes_a_false_file_absent_from_intensity():
    model = AbsoluteRefreshFiles(height=12, width=20)
    model.step(event())
    assert len(model.live_slots) == 1
    model.refresh(torch.zeros((12, 20)))
    assert not model.live_slots
    assert not bool(model.surface.confident.any())


def test_refresh_retains_a_genuine_quiet_shape():
    model = AbsoluteRefreshFiles(height=12, width=20)
    model.step(event())
    contrast = torch.zeros((12, 20))
    contrast[5, 5:7] = 1.
    model.refresh(contrast)
    assert len(model.live_slots) == 1
    assert bool(model.surface.confident[5, 5])
