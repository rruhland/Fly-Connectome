"""Distributed local state and local delayed-credit boundaries."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from distributed_predictive_sheet import DistributedPredictiveSheet
from run_distributed_predictive_sheet import occupied_sites


def sheet():
    model = DistributedPredictiveSheet(fast_units=1, units=2,
                                       height=16, width=16)
    model.input_templates.zero_()
    model.output_templates.zero_()
    model.input_templates[0, 0, 3, 3] = 1.
    return model


def pulse(*locations):
    result = torch.zeros((1, 16, 16))
    for y, x in locations:
        result[0, y, x] = 1.
    return result


def test_separate_supported_sites_activate_without_a_global_winner_cap():
    model = sheet()
    state = model.step(pulse((4, 4), (12, 12)))
    assert state[0, 1, 1] > 0.
    assert state[0, 3, 3] > 0.
    assert state[:, 0, 0].sum() == 0.


def test_delayed_signed_error_updates_only_locally_eligible_output():
    model = sheet()
    observed = torch.cat((pulse((8, 8)), torch.zeros((1, 16, 16))))
    activity = model.evidence(observed)
    target = -pulse((8, 9))
    model.credit_pair(observed, target, activity)
    assert model.output_templates[0, 0, 5, 6] < 0.
    assert model.output_templates[1].sum() == 0.


def test_local_recurrent_credit_favors_confirmed_displacement():
    model = sheet()
    previous = torch.zeros((2, 4, 4))
    current = torch.zeros_like(previous)
    previous[0, 2, 2] = 1.
    current[1, 2, 3] = 1.
    before = model.predict_state(previous)
    model.credit_transition(previous, current)
    after = model.predict_state(previous)
    assert after[1, 2, 3] > before[1, 2, 3]
    assert after[0, 2, 2] < before[0, 2, 2]


def test_quiet_frame_decays_without_inventing_new_state():
    model = sheet()
    model.step(pulse((8, 8)))
    before = model.state.clone()
    model.step(torch.zeros((1, 16, 16)))
    assert torch.allclose(model.state, before*model.decay)


def test_spatial_occupancy_uses_total_distributed_activity():
    state = torch.zeros((2, 4, 4))
    state[:, 1, 1] = .3
    assert occupied_sites(state)[1, 1]
    assert not occupied_sites(state)[0, 0]
