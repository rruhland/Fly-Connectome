"""The Pong camera adapter preserves the existing event and time semantics."""

import sys
from pathlib import Path

import torch

from fly_connectome.sensor import EventCamera

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_pong_vision import dense_events, physics_ticks_for_sample


def test_dense_events_match_existing_event_camera():
    camera = EventCamera(1, 64, 64)
    previous = torch.zeros(64, 64, dtype=torch.bool)
    frames = []
    for left in (4, 7, 7, 11):
        frame = torch.zeros(64, 64, dtype=torch.bool)
        frame[10:13, left:left+3] = True
        frames.append(frame)
    for frame in frames:
        sparse = camera.observe(frame[None])
        reference = torch.zeros(2, 64, 64)
        for pixel, on in zip(sparse.pixels.tolist(), sparse.on.tolist()):
            reference[int(on), pixel//64, pixel%64] = 1
        assert torch.equal(dense_events(previous, frame), reference)
        previous.copy_(frame)


def test_physics_ticks_preserve_120_to_50_hz_cadence():
    assert [physics_ticks_for_sample(sample) for sample in range(5)] == [2, 2, 3, 2, 3]
    assert sum(physics_ticks_for_sample(sample) for sample in range(50)) == 120
