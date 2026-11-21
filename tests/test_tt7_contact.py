"""Exercise the camera-derived credit and resumable timing model."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('mujoco')

from scripts.experimental.tt5_player import _prediction
from scripts.experimental.tt6_body_control import CameraBody
from scripts.experimental.tt6_contact import ContactPlayer
from scripts.experimental.tt7_contact import ContactTimingMemory, ContactTimingPlayer, TIMINGS, visual_result


def test_timing_shape_moves_same_stroke_through_the_estimated_contact():
    _, early, aligned, late = TIMINGS
    assert early.offset(0.) > aligned.offset(0.) > late.offset(0.)
    assert early.offset(1.) == aligned.offset(0.) == late.offset(-1.)
    assert early.offset(-10.) == aligned.offset(-10.) == late.offset(-10.)


def test_timing_explores_and_restores_contextual_preference():
    memory = ContactTimingMemory()
    cues = np.array([1., .3, -.5, .1, .2])
    for candidate in range(len(TIMINGS)):
        assert memory.choose(cues) == candidate
        memory.observe(candidate, cues, 1. if candidate == 2 else .05, 1.)
    assert memory.choose(cues, exploration=False) == 2

    restored = ContactTimingMemory.from_state(memory.state())
    assert restored.state() == memory.state()
    assert restored.choose(cues) == memory.choose(cues)
    restored.observe(2, cues, .8, .5)
    assert restored.counts[2] == memory.counts[2] + 1


def test_departure_fit_needs_coherent_stereo_evidence():
    trace = [dict(tick=tick, ball=None) for tick in range(4)]
    position, velocity = np.array([.04, -1.37, .27]), np.array([0., .135, .078])
    for tick in range(4, 16):
        trace.append(dict(tick=tick, ball=position.tolist()))
        position, velocity, _ = _prediction(position, velocity, 1., 0.)
    result = visual_result(trace, 4.)
    assert result is not None
    assert result['quality'] > .65
    assert result['departure_speed'][0] > 0.
    assert result['sightings'][0] >= 4
    for row in trace:
        if row['ball'] is not None:
            row['ball'][1] = -2.7 - row['ball'][1]
    assert visual_result(trace, 4.) is None


def test_online_finish_updates_only_the_enabled_contact_memory(monkeypatch):
    monkeypatch.setattr(ContactPlayer, 'finish', lambda self, reward: {})
    axes = 6
    fit = CameraBody(np.zeros((1 + axes + axes * (axes + 1) // 2, 4)), np.zeros(axes), np.ones((axes, 14)),
                     .01, 200, 100, True)
    prior = SimpleNamespace(choose=lambda: (0, 0))
    for explore, expected in [(True, 1), (False, 0)]:
        player = ContactTimingPlayer(fit, prior, object, explore=explore)
        player.receives.append(dict(expected=12., profile=2, cues=[1., 0., 0., 0., 0.]))
        result = player.finish(1.)
        assert player.contact_memory.counts[2] == expected
        assert result['contact_timing'][0]['score'] == 1.


def test_timed_rate_keeps_actuator_and_excursion_limits(monkeypatch):
    from scripts.experimental.tt6_body_control import MOTOR_BUDGET

    monkeypatch.setattr(ContactPlayer, 'update', lambda self, frames, preceding: np.array([.8, .95]))
    coefficients = np.zeros((6, 4))
    coefficients[1, 1:3] = [.030, .040]
    coefficients[2, 1:3] = [.040, .025]
    fit = CameraBody(coefficients, np.zeros(2), np.ones((2, 14)), .01, 200, 100, True)
    player = ContactTimingPlayer(fit, SimpleNamespace(choose=lambda: (0, 0)), object, family='rate')
    player.receives.append(dict(profile=2, rate_start=9, shaped_frames=0))
    player.servo.requested[0] = MOTOR_BUDGET - .2
    player.scene.tick = 9
    action = player.update(None)
    np.testing.assert_allclose(action, [.2, 1.])
    assert player.receives[0]['shaped_frames'] == 1
    player.scene.tick = 11
    np.testing.assert_allclose(player.update(None), [.8, .95])
