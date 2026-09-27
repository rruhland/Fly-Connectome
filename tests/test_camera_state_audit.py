"""Changed-camera audit must preserve the requested sensor geometry."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from camera_state_audit import crossing_scene


def test_square_camera_events_and_images_have_matching_geometry():
    images, events, truth = crossing_scene('vertical', 0, False)
    assert images[0].shape == (64, 64)
    assert all(event.shape == (2, 64, 64) for event in events)
    # The second entity starts in the lower half; its first event must survive.
    assert events[0][1, 54, 32] == 1
    assert torch.equal(truth[0][1], torch.tensor([54., 32.]))
