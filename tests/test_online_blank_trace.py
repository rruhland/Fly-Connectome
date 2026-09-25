"""Continuous local trace can emit while the current input is blank."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from online_blank_trace import (ContinuousTraceReadout, training_cases,
                                update_trace_weights)


def test_continuous_trace_emits_without_current_sensory_event():
    class FixedCode:
        class Motion:
            units = 24
        motion = Motion()

        def reset_state(self):
            self.trace = torch.zeros((24, 32, 64))

        def step(self, events, coincidence):
            self.trace[0, 16, 32] = 1

    model = ContinuousTraceReadout(FixedCode())
    model.weights[0, 0, 8, 8] = 1
    blank = torch.zeros((2, 32, 64))
    coincidence = torch.zeros((16, 32, 64))
    assert model.step(blank, coincidence)[0, 16, 32] == 1


def test_trace_update_uses_delayed_local_error():
    weights = torch.zeros((2, 24, 17, 17))
    error = torch.zeros((2, 32, 64))
    error[1, 16, 33] = 1
    update_trace_weights(weights, [(2, 16, 32, 1.)], error, .5)
    assert weights[1, 2, 8, 9] == .5
    assert weights.sum() == .5


def test_readout_credits_previous_blank_frame_when_event_arrives():
    class FixedCode:
        class Motion:
            units = 24
        motion = Motion()

        def reset_state(self):
            self.trace = torch.zeros((24, 32, 64))

        def step(self, events, coincidence):
            self.trace[0, 16, 32] = 1

    model = ContinuousTraceReadout(FixedCode())
    blank = torch.zeros((2, 32, 64))
    future = torch.zeros_like(blank)
    future[0, 16, 33] = 1
    coincidence = torch.zeros((16, 32, 64))
    model.step(blank, coincidence, learn=True)
    model.step(future, coincidence, learn=True)
    assert model.weights[0, 0, 8, 9] == .5


def test_training_exposes_three_gap_lengths_without_shape_labels():
    cases = training_cases()
    assert len(cases) == 288
    assert sum(family == 'clean' for family, _ in cases) == 96
    assert {family for family, _ in cases} == {
        'clean', 'early_2', 'familiar_3', 'late_4'}
    assert all(sum(family == window for family, _ in cases) == 64
               for window in ('early_2', 'familiar_3', 'late_4'))
