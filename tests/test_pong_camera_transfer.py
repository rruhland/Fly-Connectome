"""Pong supplies camera events only to the generic latent evaluator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_pong_camera_transfer import pong_events


def test_pong_camera_sampling_returns_only_event_maps():
    for stride in (1, 4):
        events = pong_events(1101, stride=stride, frames=6)
        assert len(events) == 6
        assert all(event.shape == (2, 32, 64) for event in events)
