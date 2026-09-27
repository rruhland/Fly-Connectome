"""Local associations acquire sensory selectivity without named cue classes."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from associative_patch_state import PatchAssociation, AssociativePatchState


def test_covariance_memory_learns_predictive_feature_ignoring_shared_context():
    memory = PatchAssociation(dimensions=3)
    for _ in range(4):
        memory.observe(torch.tensor([1., 0., 10.]), torch.tensor([-1., 0.]))
        memory.observe(torch.tensor([0., 1., 10.]), torch.tensor([1., 0.]))
    assert memory.predict(torch.tensor([1., 0., 10.]))[0] < -.9
    assert memory.predict(torch.tensor([0., 1., 10.]))[0] > .9
    assert memory.predict(torch.tensor([1., 0., 100.]))[0] < -.9


def test_reset_forgets_scene_but_preserves_learned_associations():
    model = AssociativePatchState(height=16, width=16)
    model.memory.observe(torch.zeros(51), torch.ones(2))
    saved = model.memory.predict(torch.zeros(51))
    model.step(torch.zeros(12, 16, 16))
    model.reset_state()
    assert torch.equal(model.memory.predict(torch.zeros(51)), saved)
    assert model.tracker.slots == []


def test_missing_camera_sample_does_not_become_negative_visual_evidence():
    model = AssociativePatchState(height=16, width=32)
    for x in range(3, 8):
        state = torch.zeros(12, 16, 32)
        state[1, 8, x] = 1.
        state[6, 8, x] = 1.
        model.step(state)
    field = model.step(torch.zeros(12, 16, 32), observation_available=False)
    assert field[8, 8] > .5
    contradicted = model.step(torch.zeros(12, 16, 32), observation_available=True)
    assert contradicted[8, 9] < .5


def test_unavailable_stale_sample_cannot_reconfirm_a_track():
    model = AssociativePatchState(height=16, width=32)
    for x in (3, 4, 5):
        sensory = torch.zeros(12, 16, 32)
        sensory[1, 8, x] = sensory[6, 8, x] = 1.
        model.step(sensory)
    model.step(sensory, observation_available=False)
    assert model.tracker.slots[0]['last_seen'] == 2
    assert torch.equal(model.entities[0]['position'], torch.tensor([8., 6.]))
