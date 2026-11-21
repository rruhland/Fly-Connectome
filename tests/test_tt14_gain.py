from types import SimpleNamespace

import numpy as np
import pytest

from scripts.experimental.tt6_body_control import CameraBody
from scripts.experimental.tt6_timing import TimingMemory
from scripts.experimental.tt8_service_drive import ServiceDriveServo
from scripts.experimental.tt13_live_ports import effective
from scripts.experimental.tt14_gain_hardware import GainWiring, perturbation
from scripts.experimental.tt14_gain_model import MotorMap, collect, fit_transform
from scripts.experimental.tt14_gain_servo import GainServo


def camera_body(width):
    coefficients = np.zeros((1 + width + width * (width + 1) // 2, 4))
    coefficients[0] = [.03, -1.2, .45, .27]
    coefficients[1:5] = np.diag([.10, .09, .11, .35])
    if width == 5:
        coefficients[5] = [.06, .013, .021, .04]
    coefficients[1 + width:] = np.random.default_rng(4793).normal(0., .002, coefficients[1 + width:].shape)
    kernels = 1. - np.exp(-np.outer(np.linspace(.28, .85, width), np.arange(14) + .4))
    return CameraBody(coefficients, np.ones(width), kernels, 0., 0, 0, True)


@pytest.mark.parametrize(('width', 'kind'), [(4, 'gain'), (5, 'coupling')])
def test_transform_predicts_rgb_effect_for_an_unseen_action_sequence(width, kind):
    base = camera_body(width)
    actual = perturbation(width, 58214071, kind)
    identity = np.eye(width)
    rng = np.random.default_rng(378417)
    original = MotorMap(base, identity, 'original')

    def observe():
        commands = rng.uniform(-.75, .75, (180, width))
        moves = np.vstack([np.zeros(width), commands])
        effective_position = effective(moves @ actual.T, base.kernels)
        poses = base.pose(effective_position[np.maximum(np.arange(len(moves)) - 2, 0)])
        trace = [dict(blade=pose[:3], blade_pitch=pose[3]) for pose in poses]
        return collect(commands, trace, width)

    candidate = fit_transform(base, [observe()], identity, kind)
    held = observe()
    assert candidate is not None
    assert max(candidate.errors(held)) < .25 * min(original.errors(held))


def test_identity_transform_preserves_the_inherited_forecast_and_strike():
    base = camera_body(4)
    first = ServiceDriveServo(base, TimingMemory())
    second = GainServo(MotorMap(base, np.eye(4), 'coupling'), TimingMemory(), learn=True)
    rng = np.random.default_rng(87364)
    for step in range(22):
        command = rng.uniform(-.8, .8, 4)
        first.advance(command)
        second.advance(command)
        pose = base.pose(first.nominal_history[0])
        actor = SimpleNamespace(position=pose[:3], observed=True)
        first.observe(actor, pose[3])
        second.observe(actor, pose[3])
        if step % 2:
            first.drive = second.drive = .132
            first.intercept = second.intercept = 3.7
        else:
            first.drive = second.drive = None
            first.intercept = second.intercept = None
        target = pose[:3] + rng.normal(0., .04, 3)
        np.testing.assert_allclose(first.command(target, target, pose[3]),
                                   second.command(target, target, pose[3]), atol=1e-12)
        np.testing.assert_allclose(first.effective, second.effective, atol=1e-12)


def test_hardware_mixes_rates_before_the_original_signed_port_map():
    arena = SimpleNamespace(action_size=4, step=lambda action, *, render: (action, render))
    matrix = perturbation(arena.action_size, 58214071, 'coupling')
    wiring = GainWiring(arena, 1067440697, matrix)
    command = np.array([.5, .24, -.3, .72])
    expected = np.empty_like(command)
    expected[wiring.order] = wiring.signs * (matrix @ command)
    applied, rendered = wiring.step(command, render=False)
    np.testing.assert_array_equal(applied, expected)
    assert not rendered
