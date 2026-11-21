from types import SimpleNamespace

import numpy as np
from scipy.signal import lfilter

import pytest

mujoco = pytest.importorskip('mujoco')

from scripts.experimental.tt6_body_arena import BODY_AXES, TT6Arena, TT6Wiring
from scripts.experimental.tt6_body_control import CameraServo, _response_bank, fit_camera_body
from scripts.experimental.tt6_body_probe import probe_commands


def test_physical_receiver_axes_and_sweep_change_the_racket_pose():
    gantry = TT6Arena(body='gantry', render_mode='projected')
    offset = TT6Arena(body='offset', render_mode='projected')
    try:
        assert gantry.action_size == 4
        assert offset.action_size == 5
        assert all(gantry.model.joint(f'learner_{axis}').type[0] == mujoco.mjtJoint.mjJNT_SLIDE
                   for axis in ('lateral', 'depth', 'lift'))
        sweep = offset.model.joint('learner_sweep')
        assert sweep.type[0] == mujoco.mjtJoint.mjJNT_HINGE
        assert abs(sweep.axis[1]) > .1
        wiring = TT6Wiring(offset, 7)
        physical = BODY_AXES['offset'][0].index('sweep')
        channel = np.flatnonzero(wiring.order == physical)[0]
        command = np.zeros(5)
        command[channel] = wiring.signs[channel]
        initial = offset.paddle.copy()
        for _ in range(5):
            wiring.step(command, render=False)
        assert offset.paddle[0] - initial[0] > .15
        assert offset.joints[physical] > .5
    finally:
        gantry.close()
        offset.close()


def test_camera_fit_uses_five_signed_channels_and_can_remove_one():
    class PoseTracker:
        def update(self, camera_pose):
            self.actor = SimpleNamespace(position=camera_pose[:3], observed=True)
            self.actor_pitch = camera_pose[3]

    response_bank, _ = _response_bank()
    responses = response_bank[[20, 31, 44, 34, 39]]
    home = np.array((.04, -1.44, .22, .29))
    motor_map = np.array(((.057, 0., .03, 0.), (-.075, 0., .02, .01), (0., 0., 0., -.17),
                          (.01, -.075, .02, 0.), (0., .018, .084, 0.)))
    histories = []
    for seed in range(7):
        actions = np.vstack((np.zeros(5), probe_commands(seed, 34, 5)))
        filtered = np.cumsum(actions, axis=0) + np.column_stack(
            [lfilter(kernel - 1, [1.], action) for kernel, action in zip(responses, actions.T, strict=True)])
        histories.append(list(zip(actions, home + filtered @ motor_map, strict=True)))
    fit = fit_camera_body(histories, PoseTracker)
    assert fit.action_size == 5
    assert fit.pitch_controllable
    assert fit.error < .006
    np.testing.assert_allclose(fit.jacobian(np.zeros(5)).T, motor_map, atol=.012)
    modified = fit.without_channel(1)
    np.testing.assert_array_equal(modified.jacobian(np.ones(5))[:, 1], 0.)
    servo = CameraServo(modified)
    command = servo.command(fit.ready + (.15, .03, -.03), fit.ready + (.16, .04, -.04), .47)
    assert command.shape == (5,)
    assert abs(command[1]) < 1e-10
    np.testing.assert_allclose(fit.reversed_channels().pose(np.ones(5)), fit.pose(-np.ones(5)))
