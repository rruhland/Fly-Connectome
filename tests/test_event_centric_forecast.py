"""Local competitive forecast learns only from arrived future events."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from event_centric_forecast import (CompetitiveLocalForecast,
                                    MotionWithSeparateSurface)


def state(*entries):
    value = torch.zeros((2, 32, 64))
    for unit, y, x, amplitude in entries:
        value[unit, y, x] = amplitude
    return value


def event(channel=1, y=12, x=20):
    value = torch.zeros((2, 32, 64))
    value[channel, y, x] = 1
    return value


def test_only_one_local_unit_wins_each_retinotopic_site():
    head = CompetitiveLocalForecast(2, horizon=1)
    chosen = head.sources(state((0, 12, 20, .8), (1, 12, 20, .9),
                                (0, 12, 21, .4)))
    assert len(chosen) == 1
    assert chosen[0][:3] == (12, 20, 1)
    assert chosen[0][3] == pytest.approx(.9)


def test_four_frame_local_error_waits_until_target_arrives():
    head = CompetitiveLocalForecast(2, horizon=4, eta=1)
    head.step(state((0, 12, 20, 1)), event(), learn=True)
    for _ in range(3):
        head.step(state(), torch.zeros_like(event()), learn=True)
    assert head.weights.sum() == 0
    head.step(state(), event(y=13), learn=True)
    assert head.weights[0, 1, 3, 2] == 1
    head.reset_state()
    forecast = head.step(state((0, 12, 20, 1)), event())
    assert forecast[1, 13, 20] == 1


def test_quiet_credit_reduces_local_future_event_probability():
    head = CompetitiveLocalForecast(2, horizon=1, eta=.5)
    head.weights[0, 1, 3, 2] = 1
    head.step(state((0, 12, 20, 1)), event(), learn=True)
    head.step(state(), torch.zeros_like(event()), learn=True)
    assert head.weights[0, 1, 3, 2] == .5


def test_overlapping_local_sources_average_instead_of_summing():
    head = CompetitiveLocalForecast(2, horizon=1)
    head.weights[0, 1, 2, 2] = 1
    head.weights[1, 1, 2, 1] = 1
    prediction = head.step(state((0, 12, 20, 1), (1, 12, 21, 1)), event())
    assert prediction[1, 12, 20] == 1
    assert prediction.max() == 1


def test_next_event_credit_waits_through_quiet_frames():
    head = CompetitiveLocalForecast(2, next_event=True, eta=1)
    head.step(state((0, 12, 20, 1)), event(), learn=True)
    assert head.step(state(), torch.zeros_like(event()), learn=True) is None
    assert head.weights.sum() == 0
    head.step(state(), event(y=13), learn=True)
    assert head.weights[0, 1, 3, 2] == 1


def test_persistent_surface_is_kept_separate_from_event_centric_latent():
    class FakeMotion:
        units = 2

        def reset_state(self):
            self.latent = torch.zeros((2, 32, 64))

        def step(self, code):
            self.latent = code.clone()

    encoder = MotionWithSeparateSurface(FakeMotion())
    surface = torch.ones((2, 32, 64))
    code = state((0, 12, 20, 1))
    encoder.step((code, surface))
    assert torch.equal(encoder.latent, code)
    assert torch.equal(encoder.observation, surface)
