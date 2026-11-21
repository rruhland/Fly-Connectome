
import numpy as np
import pytest

pytest.importorskip('mujoco')
from scripts.experimental.tt5_player import GRAVITY, _launch_pitch
from scripts.experimental.tt7_flight import FlightMemory, nominal_exit, outgoing_speed


def test_untrained_impulse_starts_with_same_shot_and_updated_state_restores():
    point, incoming = np.array([.12, -1.36, .28]), np.array([.011, -.08, .023])
    memory = FlightMemory('impulse')
    pitch = memory.choose_angle(point, incoming, .065, .80)
    assert pitch == _launch_pitch(point, incoming, .065, .80)
    proposed = nominal_exit(incoming[1:], pitch, .065, .8)
    for _ in range(5):
        memory.sample((incoming[1:], pitch, .065), proposed - [.052, .014], proposed)
    restored = FlightMemory('impulse', memory.history, memory.impulses)
    assert restored.choose_angle(point, incoming, .065, .80) == memory.choose_angle(point, incoming, .065, .8)
    assert memory.choose_angle(point, incoming, .065, .80) != pitch


def test_camera_departure_requires_coherent_observed_motion():
    def path(sign):
        return [dict(tick=t, ball=[.1, -1.39 + sign*.09*t, .22 + .08*t - .5*GRAVITY*t*t])
                for t in range(1, 7)]
    assert outgoing_speed(path(-1), 0.) is None
    speed, _, samples = outgoing_speed(path(1), 0.)
    np.testing.assert_allclose(speed, [.09, .08], atol=.007)
    assert len(samples) == 4


def test_frozen_memory_ignores_evidence():
    memory = FlightMemory('adapt', [[-.04, .01]], frozen=True)
    memory.sample((np.array([-.09, .02]), .2, .04), np.ones(2), np.zeros(2))
    assert list(memory.history) == [[-.04, .01]]
    assert not memory.impulses
