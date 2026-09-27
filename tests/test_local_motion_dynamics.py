"""Temporal local associations learn transitions without mutating at inference."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_motion_dynamics import LocalMotionDynamics
from run_local_motion_dynamics import visual_episode
from visual_history_benchmark import SHAPES


def test_alternating_motion_is_learned_and_rotation_transfers():
    model = LocalMotionDynamics()
    first = torch.tensor([[0., 1.], [0., 3.], [0., 1.], [0., 3.]])
    second = torch.tensor([[0., 3.], [0., 1.], [0., 3.], [0., 1.]])
    for _ in range(4):
        model.observe(first, torch.tensor([0., 1.]))
        model.observe(second, torch.tensor([0., 3.]))
    assert torch.allclose(model.predict(first), torch.tensor([0., 1.]), atol=.1)
    rotated = first.flip(1)
    assert torch.allclose(model.predict(rotated), torch.tensor([1., 0.]), atol=.1)
    saved = model.values.clone()
    model.rollout(first, 8)
    assert torch.equal(model.values, saved)


def test_rollout_has_no_access_to_future_observations_and_stays_finite_at_rest():
    model = LocalMotionDynamics()
    assert torch.equal(model.rollout(torch.zeros(4, 2), 8), torch.zeros(2))


def test_asymmetric_motif_truth_is_its_geometric_center(monkeypatch):
    monkeypatch.setitem(SHAPES, 'asymmetric', ((0, 0), (1, 0), (1, 1)))
    case, truth = visual_episode('constant', 0, 'asymmetric')
    assert torch.allclose(truth[0], case['visible'][0].nonzero().float().mean(0))
