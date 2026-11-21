"""Protect the learned servo against stale model predictions and bad sightings."""

from types import SimpleNamespace

import numpy as np

import pytest

pytest.importorskip('mujoco')

from scripts.experimental.tt6_body_control import CameraBody
from scripts.experimental.tt6_rotary_adapt import LagAwareServo, StrokeMemory
from scripts.experimental.tt6_timing import TimingMemory


def test_camera_innovation_survives_model_drift_after_a_coherent_sighting():
    axes = 6
    coefficients = np.zeros((1 + axes + axes * (axes + 1) // 2, 4))
    coefficients[0, :3] = [.08, -1.40, .28]
    fit = CameraBody(coefficients, np.zeros(axes), np.ones((axes, 14)), .02, 100, 100, True)
    servo = LagAwareServo(fit, TimingMemory())
    sighting = SimpleNamespace(position=(.08, -1.40, .05), observed=True)
    servo.observe(sighting, None)
    assert servo.pose[2] == .28
    servo.observe(sighting, None)
    assert servo.pose[2] < .16
    first = servo.pose[2]
    servo.observe(SimpleNamespace(position=(.08, -1.4, .70), observed=True), None)
    assert servo.pose[2] == first


def test_stroke_memory_uses_only_the_visible_departure_and_public_reward():
    memory = StrokeMemory()
    for choice in range(3):
        assert memory.choose(explore=True) == choice
        flight = {'landing': [.09, .40], 'net_height': .23} if choice == 1 else None
        memory.observe(choice, flight, -1.)
    assert memory.choose(explore=False) == 1
    assert memory.state()['counts'] == [1, 1, 1]
