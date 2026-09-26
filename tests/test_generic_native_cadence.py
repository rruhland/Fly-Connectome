"""Native-cadence training scenes use only generic visual primitives."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from generic_native_cadence import scene_events
from run_absolute_refresh import contrast_frames


def test_scene_events_are_deterministic_signed_camera_observations():
    first = scene_events(17, frames=24)
    again = scene_events(17, frames=24)
    assert len(first) == 24
    assert all(event.shape == (2, 32, 64) for event in first)
    assert all(torch.equal(a, b) for a, b in zip(first, again))
    assert sum(float(event.sum()) for event in first) > 0
    assert any(event.sum() == 0 for event in first)


def test_heldout_scene_is_a_distinct_generic_sequence():
    train = scene_events(7, frames=24)
    heldout = scene_events(7, frames=24, heldout=True)
    assert any(not torch.equal(a, b) for a, b in zip(train, heldout))


def test_optional_intensity_is_rendered_independently_of_camera_events():
    events, intensity = scene_events(7, frames=24, return_contrast=True)
    assert len(events) == len(intensity) == 24
    assert all(torch.equal(a, b) for a, b in zip(
        intensity, contrast_frames(events)))


def test_camera_stride_changes_unlabeled_temporal_observations():
    slow = scene_events(7, frames=24, motion_stride=1)
    fast = scene_events(7, frames=24, motion_stride=4)
    assert len(slow) == len(fast) == 24
    assert any(not torch.equal(a, b) for a, b in zip(slow, fast))
