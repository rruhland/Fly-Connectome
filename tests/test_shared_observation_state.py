"""Two histories must not be overwritten by one ambiguous visible region."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from camera_state_audit import crossing_scene
from shared_observation_state import SharedObservationState


def test_identity_survives_a_visible_merge_after_missing_camera_samples():
    images, events, truth = crossing_scene('horizontal', -4, False)
    model = SharedObservationState(height=64, width=64)
    identities = []
    for frame, (image, event) in enumerate(zip(images, events)):
        available = not 16 <= frame < 21
        sensory = torch.zeros(12, 64, 64)
        if available:
            sensory[:2] = event
            sensory[6] = image
        model.step(sensory, observation_available=available)
        if frame == 6:
            identities = [min(model.entities,
                              key=lambda e: float((e['position']-target).norm()))['id']
                          for target in truth[frame]]
    entities = {e['id']: e for e in model.entities}
    assert len(set(identities)) == 2
    for identity, target in zip(identities, truth[-1]):
        assert torch.allclose(entities[identity]['position'], target, atol=1.)
