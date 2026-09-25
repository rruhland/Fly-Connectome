"""Causal local credit for fixed-frame and next-event visual forecasts."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from delayed_visual_emission import (DelayedLocalEmission,
                                     RecentObservationSurface,
                                     SurfaceMotionEncoder)


def impulse(channel=1, y=12, x=20):
    value = torch.zeros((2, 32, 64))
    value[channel, y, x] = 1
    return value


def state(y=12, x=20):
    value = torch.zeros((3, 32, 64))
    value[0, y, x] = 1
    return value


def test_four_frame_credit_arrives_only_at_due_frame():
    learner = DelayedLocalEmission(3, horizon=4, eta=1)
    first = learner.step(state(), impulse(), learn=True)
    assert first.sum() == 0
    for _ in range(3):
        learner.step(torch.zeros_like(state()), torch.zeros_like(impulse()),
                     learn=True)
    assert learner.weights.sum() == 0
    learner.step(torch.zeros_like(state()), impulse(y=13), learn=True)
    assert learner.weights[1, 0, 3, 2] > 0
    assert learner.weights[1, 0, 2, 2] == 0


def test_next_event_credit_waits_through_quiet_and_resets_episode():
    learner = DelayedLocalEmission(3, next_event=True, eta=1)
    learner.step(state(), impulse(), learn=True)
    for _ in range(5):
        assert learner.step(state(), torch.zeros_like(impulse()),
                            learn=True) is None
    assert learner.weights.sum() == 0
    learner.step(state(), impulse(y=13), learn=True)
    assert learner.weights[1, 0, 3, 2] > 0
    learner.reset_state()
    before = learner.weights.clone()
    learner.step(state(), impulse(y=14), learn=True)
    assert torch.equal(learner.weights, before)


def test_shuffled_credit_uses_supplied_target_only_when_due():
    learner = DelayedLocalEmission(3, horizon=1, eta=1)
    learner.step(state(), impulse(), learn=True)
    learner.step(torch.zeros_like(state()), impulse(y=13), learn=True,
                 credit_target=impulse(channel=0, y=11))
    assert learner.weights[0, 0, 1, 2] > 0
    assert learner.weights[1].sum() == 0


def test_surface_motion_encoder_keeps_learned_and_observed_channels_distinct():
    class FakeMotion:
        units = 1

        def reset_state(self):
            self.latent = torch.zeros((1, 32, 64))

        def step(self, primitive):
            self.latent = primitive[:1]

    encoder = SurfaceMotionEncoder(FakeMotion())
    encoder.reset_state()
    encoder.step((impulse(), impulse(channel=0)))
    assert encoder.units == 3
    assert encoder.latent[0, 12, 20] == 0
    assert encoder.latent[1, 12, 20] == 1
    assert encoder.latent[2, 12, 20] == 0


def test_recent_observation_adds_causal_decaying_local_history():
    history = RecentObservationSurface(decay=.5)
    first = history.step(impulse()).clone()
    quiet = history.step(torch.zeros_like(impulse()))
    assert first.shape == (4, 32, 64)
    assert quiet[1, 12, 20] == 1
    assert quiet[3, 12, 20] == .5
    assert quiet[2, 12, 20] == 0
    history.reset_state()
    assert history.channels.sum() == 0
