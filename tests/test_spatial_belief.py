import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from spatial_belief import SpatialBelief, MarginalCalibration


def test_identical_history_keeps_two_observed_futures():
    model = SpatialBelief()
    history = torch.tensor([[0., 1.]]*4)
    for _ in range(20):
        model.observe(history, torch.tensor([-6., 4.]))
        model.observe(history, torch.tensor([6., 4.]))
    centers, weights = model.distribution(history)
    assert (centers[:, 0] < -5).any() and (centers[:, 0] > 5).any()
    assert torch.allclose(weights.sum(), torch.tensor(1.))
    assert model.log_prob(history, torch.tensor([6., 4.])) > model.log_prob(history, torch.tensor([0., 4.]))


def test_memory_bounded_and_prediction_does_not_learn():
    model = SpatialBelief(capacity=16)
    history = torch.tensor([[0., 1.]]*4)
    for i in range(40):
        model.observe(history, torch.tensor([float(i), 4.]))
    before = model.values.clone()
    model.predict(history)
    assert len(model.keys) == 16 and model.seen == 40
    assert torch.equal(before, model.values)


def test_rotation_and_scale_transfer():
    model = SpatialBelief()
    history = torch.tensor([[0., 1.]]*4)
    model.observe(history, torch.tensor([2., 4.]))
    rotation = torch.tensor([[0., -1.], [1., 0.]])
    assert torch.allclose(model.predict(2*history @ rotation.T), torch.tensor([-8., 4.]))


def test_calibration_uses_bounded_observed_ranks():
    model = MarginalCalibration()
    for i in range(600):
        model.observe(torch.tensor([i/600., i/600.]))
    low, high = model.bounds()
    assert len(model.ranks) == 512
    assert torch.all(low >= 88/600.) and torch.all(high <= 1.)
    assert torch.all(high > low)


def test_issued_distribution_is_credited_before_memory_changes():
    model = SpatialBelief(horizon=4)
    history = torch.tensor([[0., 1.]]*4)
    mean, credit = model.predict_with_credit(history)
    assert torch.equal(mean, torch.tensor([0., 4.]))
    model.observe(history, torch.tensor([20., 20.]))
    model.observe(history, mean, credit=credit)
    assert torch.allclose(model.calibration.ranks[-1], torch.tensor([.5, .5]))
