"""Local transition and signed sensory correction."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from predictive_pair_assemblies import PredictivePairAssemblies
from temporal_pair_world_state import TemporalPairWorldState


def model():
    paired = PredictivePairAssemblies(fast_units=1, units=2,
                                      height=16, width=16, max_winners=1)
    return TemporalPairWorldState(paired)


def test_local_transition_learns_next_sparse_unit_and_displacement():
    state = model()
    previous = torch.zeros_like(state.state)
    current = torch.zeros_like(state.state)
    previous[0, 1, 1] = 1.
    current[1, 1, 2] = 1.
    before = state.predict_state(previous)
    state.credit_transition(previous, current)
    predicted = state.predict_state(previous)
    assert predicted[1, 1, 2] > predicted[1, 1, 0]
    assert predicted[1, 1, 2] > before[1, 1, 2]
    assert predicted[0, 1, 1] < before[0, 1, 1]


def test_signed_local_decoder_can_suppress_unsupported_future_activity():
    state = model()
    state.decoder.zero_()
    active = torch.zeros_like(state.state)
    active[0, 2, 2] = 1.
    error = torch.zeros((1, 16, 16))
    error[0, 8, 9] = -1.
    state.credit_decoder(active, error)
    assert state.decoder[0, 0, 5, 6] < 0.
    assert state.decode(active)[0, 8, 9] < 0.


def test_recurrent_forecast_adds_signed_local_correction_to_fast_baseline():
    state = model()
    state.decoder.zero_()
    state.decoder[0, 0, 5, 5] = -.5
    state.transitions[0, 0, 1, 1] = 1.
    state.paired.state[0, 2, 2] = 1.
    baseline = torch.zeros((1, 16, 16))
    baseline[0, 8, 8] = 1.
    dictionary = torch.zeros((1, 2, 5, 5))
    dictionary[0, 0, 2, 2] = 1.
    forecast = state.forecast(dictionary, baseline)
    assert torch.isclose(forecast[0, 8, 8], torch.tensor(.5))
