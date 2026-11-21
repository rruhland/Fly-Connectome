import numpy as np
import pytest

pytest.importorskip('mujoco')
from scripts.experimental.tt5_player import GRAVITY
from scripts.experimental.tt8_service_player import ServiceMemory, falling_crossing


def test_camera_toss_forecast_recovers_descending_intersection():
    initial = np.array([.06, -1.51, .31])
    velocity = np.array([.001, -.0005, .090])
    rows = [(t, initial + t * velocity - [0., 0., .5 * GRAVITY * t**2]) for t in (1, 3, 5)]
    duration, xy = falling_crossing(rows, initial[2])
    assert duration == pytest.approx(2. * velocity[2] / GRAVITY)
    np.testing.assert_allclose(xy, initial[:2] + duration * velocity[:2])
    assert falling_crossing(rows[:2], initial[2]) is None


def test_rewarded_service_memory_reloads_and_keeps_updating():
    state = ServiceMemory()
    first = state.choose()
    state.observe(first, 0.)
    preferred = state.choose()
    assert preferred != first
    state.observe(preferred, 1.)
    restored = ServiceMemory(state.state())
    assert restored.choose() == state.choose()
    restored.observe(restored.choose(), .25)
    assert restored.counts.sum() == 3
