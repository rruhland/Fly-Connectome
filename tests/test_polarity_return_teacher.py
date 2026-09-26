"""Delayed local observation supplies credit; future input is not inferred."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from polarity_return_teacher import PolarityReturnTeacher


def event(channel=None, y=3, x=3):
    raw = torch.zeros(2, 8, 8)
    if channel is not None:
        raw[channel, y, x] = 1.
    return raw


def test_credit_waits_for_window_and_uses_same_pixel_opposite_polarity():
    teacher = PolarityReturnTeacher(window=3)
    assert teacher.step(event(1), 'arrival') is None
    assert teacher.step(event(1), 'same sign') is None
    assert teacher.step(event(0), 'return') is None
    features, raw, target = teacher.step(event(), 'now')
    assert features == 'arrival'
    assert raw[1, 3, 3] == 1
    assert target[1, 3, 3] == 1
    assert target.sum() == 1


def test_other_pixels_cannot_validate_an_event_and_reset_discards_credit():
    teacher = PolarityReturnTeacher(window=2)
    teacher.step(event(1), None)
    teacher.step(event(0, x=4), None)
    _, _, target = teacher.step(event(), None)
    assert target.sum() == 0
    teacher.reset_state()
    assert teacher.step(event(), None) is None
