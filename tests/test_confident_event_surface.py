"""A local signed surface retains supported evidence without naming objects."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from confident_event_surface import ConfidentEventSurface


def event(*, on=(), off=()):
    result = torch.zeros((2, 12, 16))
    for y, x in on:
        result[1, y, x] = 1.
    for y, x in off:
        result[0, y, x] = 1.
    return result


def test_coherent_signed_shape_survives_quiet_frames():
    model = ConfidentEventSurface(height=12, width=16)
    model.step(event(off=((5, 6), (5, 7), (5, 8))))
    for _ in range(12):
        model.step(event())
    assert model.confident[5, 6:9].all()
    assert (model.contrast[5, 6:9] < 0).all()


def test_single_false_event_is_not_promoted_without_support():
    model = ConfidentEventSurface(height=12, width=16)
    model.step(event(on=((5, 6),)))
    assert model.contrast[5, 6] > 0
    assert not model.confident[5, 6]
    for _ in range(10):
        model.step(event())
    assert not model.confident[5, 6]


def test_repeated_local_motion_can_confirm_a_single_pixel():
    model = ConfidentEventSurface(height=12, width=16)
    model.step(event(on=((5, 6),)))
    model.step(event(on=((5, 7),), off=((5, 6),)))
    model.step(event(on=((5, 8),), off=((5, 7),)))
    assert model.confident[5, 8]
    assert not model.confident[5, 6]


def test_reset_removes_prior_surface_evidence():
    model = ConfidentEventSurface(height=12, width=16)
    model.step(event(on=((5, 6), (5, 7))))
    model.reset_state()
    assert model.contrast.sum() == 0
    assert model.confidence.sum() == 0
