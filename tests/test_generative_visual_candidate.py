import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_generative_dynamics import LocalGenerativeDynamics
from statistical_visual_state import StatisticalVisualCandidate


def test_generative_map_learns_rotation_equivariant_linear_motion():
    model = LocalGenerativeDynamics()
    history = torch.tensor([[0., 2.]]*4)
    for _ in range(30):
        model.observe(history, torch.tensor([0., 16.]))
    assert torch.allclose(model.predict(history.flip(1)), torch.tensor([16., 0.]), atol=.2)
    prior = model.coefficients.clone()
    model.predict(history)
    assert torch.equal(prior, model.coefficients)


def test_statistical_observation_admits_diagonal_motion_and_rejects_blank_noise():
    model = StatisticalVisualCandidate(height=32, width=32)
    for t in range(5):
        image = torch.zeros(32, 32)
        image[3+4*t, 3+4*t] = 1.
        result = model.step(torch.zeros(2, 32, 32), image)
    assert len(result['entities']) == 1
    assert result['entities'][0]['observed']
    model.reset_state()
    generator = torch.Generator().manual_seed(151003)
    for _ in range(30):
        result = model.step(torch.zeros(2, 32, 32), .4+.02*torch.randn(32, 32, generator=generator))
        assert not result['entities']
