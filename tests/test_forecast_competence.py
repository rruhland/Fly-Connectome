import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from contextual_motion_dynamics import ContextualMotionDynamics
from forecast_competence import ForecastCompetence


def test_credit_uses_issued_candidates_even_after_base_weights_change():
    history = torch.tensor([[0., 1.]]*4)
    base = ContextualMotionDynamics()
    base.observe(history, torch.tensor([5., 1.]))
    model = ForecastCompetence(base, 4)
    _, credit = model.predict_with_credit(history)
    expected = (credit['candidates']-torch.tensor([0., 4.])).square().sum(1)
    base.values.fill_(100.)
    model.observe(history, torch.tensor([0., 4.]), credit=credit)
    assert torch.equal(model.risk.values[-1], expected)
    assert torch.allclose(model.predict(history), torch.tensor([0., 4.]))
