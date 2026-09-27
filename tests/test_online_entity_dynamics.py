import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from online_entity_dynamics import OnlineEntityDynamics


def test_observed_learning_is_rotation_equivariant_and_missing_evidence_cannot_credit():
    a, b = OnlineEntityDynamics(), OnlineEntityDynamics()
    rotation = torch.tensor([[0., -1.], [1., 0.]])
    for t in range(12):
        position = torch.tensor([3*torch.sin(torch.tensor(t*.2)), 3*torch.cos(torch.tensor(t*.2))])
        a.observe(position)
        b.observe(position @ rotation)
    assert torch.allclose(a.predict(4) @ rotation, b.predict(4), atol=1e-4)
    prior = a.coefficients.clone()
    a.observe(None)
    assert torch.equal(a.coefficients, prior)
    assert a.predict(1) is None
