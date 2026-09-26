"""Local paired-evidence selection and inference boundaries."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from predictive_pair_assemblies import PredictivePairAssemblies


def pulse(y=8, x=8):
    field = torch.zeros((2, 16, 16))
    field[0, y, x] = 1.
    return field


def observed(y=8, x=8):
    current = pulse(y, x)
    return torch.cat((current, torch.zeros_like(current)))


def test_delayed_future_evidence_can_choose_a_different_local_unit():
    model = PredictivePairAssemblies(fast_units=2, units=2,
                                     height=16, width=16, max_winners=1)
    model.input_templates.zero_()
    model.future_templates.zero_()
    model.input_templates[:, 0, 3, 3] = 1.
    model.future_templates[0, 0, 5, 5] = 1.
    model.future_templates[1, 0, 5, 6] = 1.
    current = observed()
    left = model.select(current, pulse(), use_future=True)
    right = model.select(current, pulse(8, 9), use_future=True)
    assert left[0, 2, 2] == 1.
    assert right[1, 2, 2] == 1.
    assert model.select(current, pulse(8, 9), use_future=False).equal(
        model.select(current, pulse(), use_future=False))


def test_pair_update_changes_only_selected_templates_and_translates():
    model = PredictivePairAssemblies(fast_units=2, units=2,
                                     height=16, width=16, max_winners=1)
    model.input_templates.zero_()
    model.future_templates.zero_()
    model.input_templates[0, 0, 3, 3] = 1.
    before_other = model.input_templates[1].clone()
    chosen = model.select(observed(), None, use_future=False)
    model.credit_pair(observed(), pulse(8, 9), chosen)
    assert model.future_templates[0, 0, 5, 6] > 0.
    assert model.input_templates[1].equal(before_other)
    model.reset_state()
    translated = model.step(pulse(12, 8))
    assert translated[0, 3, 2] == 1.


def test_forecast_uses_present_state_and_does_not_consult_future_target():
    model = PredictivePairAssemblies(fast_units=2, units=1,
                                     height=16, width=16, max_winners=1)
    model.input_templates.zero_()
    model.future_templates.zero_()
    model.input_templates[0, 0, 3, 3] = 1.
    model.future_templates[0, 0, 5, 6] = .5
    model.step(pulse())
    first = model.predict_fast().clone()
    model.step(torch.zeros((2, 16, 16)))
    second = model.predict_fast()
    assert first[0, 8, 9] > 0.
    assert 0 < second[0, 8, 9] < first[0, 8, 9]
