import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_linear_dynamics import LocalLinearDynamics


def test_local_correlations_learn_shared_temporal_relation_and_rotate():
    model = LocalLinearDynamics()
    generator = torch.Generator().manual_seed(50)
    for _ in range(30):
        history = torch.randn(4, 2, generator=generator)
        model.observe(history, 2*history[-1]-history[-2])
    history = torch.randn(4, 2, generator=generator)
    assert torch.allclose(model.predict(history), 2*history[-1]-history[-2], atol=.001)
    rotation = torch.tensor([[0., -1.], [1., 0.]])
    assert torch.allclose(model.predict(history @ rotation), model.predict(history) @ rotation)
