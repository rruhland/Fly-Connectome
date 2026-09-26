"""Local learned continuity changes the live state, not just a readout."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from continuity_binding_field import ContinuityBindingField
from predictive_pair_assemblies import PredictivePairAssemblies


def field():
    paired = PredictivePairAssemblies(fast_units=1, units=2,
                                      height=16, width=16, max_winners=2)
    paired.input_templates.zero_()
    paired.input_templates[0, 0, 3, 3] = 1.
    return ContinuityBindingField(paired)


def pulse(y=8, x=8):
    result = torch.zeros((1, 16, 16))
    result[0, y, x] = 1.
    return result


def test_local_temporal_association_moves_live_state():
    model = field()
    previous = torch.zeros_like(model.state)
    current = torch.zeros_like(model.state)
    previous[0, 2, 2] = 1.
    current[1, 2, 3] = 1.
    model.credit(previous, current)
    model.normalize()
    assert model.temporal[1, 0, 2, 1] > 0.
    model.state = previous.clone()
    model.paired.input_templates.zero_()
    model.step(pulse())
    assert model.state[1, 2, 3] > 0.


def test_common_fate_creates_only_local_lateral_association():
    model = field()
    previous = torch.zeros_like(model.state)
    current = torch.zeros_like(model.state)
    previous[0, 1, 1] = current[0, 1, 1] = 1.
    previous[1, 1, 2] = current[1, 1, 2] = 1.
    model.credit(previous, current)
    model.normalize()
    assert model.lateral[1, 0, 2, 1] > 0.
    assert model.lateral[1, 0, 0, 0] == 0.


def test_quiet_observation_decays_state_without_spatial_propagation():
    model = field()
    model.state[0, 2, 2] = 1.
    model.temporal[1, 0, 2, 1] = 1.
    model.step(torch.zeros((1, 16, 16)))
    assert model.state[0, 2, 2] == model.decay
    assert model.state[1].sum() == 0.
