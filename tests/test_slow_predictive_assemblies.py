"""Opt-in slow assemblies bind spatial fast evidence with local credit."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from slow_predictive_assemblies import SlowPredictiveAssemblies


def pulse(y, x, *, channel=0):
    fast = torch.zeros((2, 16, 16))
    fast[channel, y, x] = 1.
    return fast


def centered_assembly():
    model = SlowPredictiveAssemblies(fast_units=2, units=1,
                                     height=16, width=16,
                                     max_winners=1)
    model.dictionary.zero_()
    model.dictionary[0, 0, 3, 3] = 1.
    return model


def test_shared_assembly_follows_translated_fast_pattern():
    model = centered_assembly()
    first = model.step(pulse(4, 4))
    model.reset_state()
    second = model.step(pulse(8, 12))
    assert first.sum() == 1
    assert second.sum() == 1
    assert first[0, 1, 1] == 1
    assert second[0, 2, 3] == 1


def test_local_residual_updates_shared_slow_motif():
    model = centered_assembly()
    model.dictionary[0, 0, 3, 3] = .2
    model.step(pulse(4, 4), learn_sensory=True)
    assert model.dictionary[0, 0, 3, 3] > .2
    assert model.dictionary[0, 1].sum() == 0


def test_recurrence_learns_assembly_displacement():
    model = centered_assembly()
    first = model.step(pulse(4, 4)).clone()
    model.reset_state()
    second = model.step(pulse(4, 8)).clone()
    model.credit_transition(first, second)
    predicted = model.predict_field(first)
    assert predicted[0, 1, 2] > predicted[0, 1, 0]


def test_quiet_frame_preserves_decaying_assembly_context():
    model = centered_assembly()
    model.step(pulse(4, 4))
    before = model.state.clone()
    model.step(torch.zeros((2, 16, 16)))
    assert 0 < model.state[0, 1, 1] < before[0, 1, 1]
