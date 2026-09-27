"""Online forecasts cannot learn from observations that have not arrived."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_motion_dynamics import LocalMotionDynamics
from run_local_motion_dynamics import forecast_positions
from run_direct_dynamics import forecast


def test_future_observation_changes_cannot_affect_earlier_online_forecasts():
    first = [torch.tensor([0., float(t)]) for t in range(20)]
    second = [p.clone() if t <= 8 else p+torch.tensor([10., -4.])
              for t, p in enumerate(first)]
    models = dict(learned=LocalMotionDynamics(), shuffled=LocalMotionDynamics())
    a = forecast_positions(models, first, online=True)
    b = forecast_positions(models, second, online=True)
    for key in a:
        if key[0] <= 8:
            assert torch.equal(a[key]['learned'], b[key]['learned'])
            assert torch.equal(a[key]['shuffled'], b[key]['shuffled'])
    assert len(models['learned'].keys) == 0


def test_direct_delayed_credit_is_prefix_causal_and_preserves_base_weights():
    first = [torch.tensor([0., float(t)]) for t in range(20)]
    second = [p.clone() if t <= 8 else p+torch.tensor([10., -4.])
              for t, p in enumerate(first)]
    models = {name: {h: LocalMotionDynamics() for h in (1, 4, 8)}
              for name in ('learned', 'shuffled')}
    a, b = forecast(models, first, online=True), forecast(models, second, online=True)
    for key in a:
        if key[0] <= 8:
            for name in ('learned', 'shuffled'):
                assert torch.equal(a[key][name], b[key][name])
    assert all(len(model.keys) == 0 for arm in models.values() for model in arm.values())
