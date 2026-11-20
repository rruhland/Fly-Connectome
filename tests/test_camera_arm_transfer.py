"""Public RGB sensing, delayed drives, and paired controls for the camera-arm task."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from body_reaching_arena import SUCCESS_TICKS
from body_strategy_transfer import CueLayout
from camera_motor_learner import CALIBRATION_TICKS, CameraMotorLearner
from camera_reaching_arena import CameraArena, CameraView, Drive, decode_camera, render_camera
from run_camera_arm_transfer import MOTOR, _ranked_goals, run


LAYOUT = CueLayout((.38, -.08), ((.30, -.30), (.57, .27)), (-.26, .01))


def test_rgb_detections_follow_private_camera_projection_and_preserve_distance_order():
    rng = np.random.default_rng(1139)
    fiducials = ((-.61, -.46), (-.63, .37), (.72, -.49))
    for tick in (0, 19, 31):
        view = CameraView.sample(rng, changed=True)
        frame = render_camera(view, tick, LAYOUT.start, LAYOUT.markers, LAYOUT.reference, fiducials=fiducials)
        observed = decode_camera(frame)
        matrix, offset = view.at(tick)
        np.testing.assert_allclose(observed.tip, matrix @ LAYOUT.start + offset, atol=.55)
        np.testing.assert_allclose(observed.reference, matrix @ LAYOUT.reference + offset, atol=.55)
        np.testing.assert_allclose(observed.fiducials, np.asarray(fiducials) @ matrix.T + offset, atol=.55)
        np.testing.assert_allclose(_ranked_goals(observed), np.asarray(LAYOUT.by_distance()) @ matrix.T + offset, atol=.55)


def test_rgb_is_independent_of_reward_and_positions_move_only_after_actuation_delay():
    rng = np.random.default_rng(7762)
    view = CameraView.sample(rng)
    drive = Drive(2, tuple(map(tuple, np.eye(3))), (.64,) * 3, (1.6,) * 3)
    near = CameraArena(LAYOUT, 0, view, max_steps=20, drive=drive)
    far = CameraArena(LAYOUT, 1, view, max_steps=20, drive=drive)
    np.testing.assert_array_equal(near.frame(), far.frame())
    for command in ((1., 0., 0.), (0., 0., 0.), (0., 0., 0.)):
        near_frame, near_reward = near.step(command)
        far_frame, far_reward = far.step(command)
        np.testing.assert_array_equal(near_frame, far_frame)
        assert near_reward == far_reward == 0
    np.testing.assert_allclose(near.trace[:3], [LAYOUT.start] * 3, atol=1e-12)
    assert np.linalg.norm(np.subtract(near.trace[-1], LAYOUT.start)) > .01


def test_contact_reward_requires_true_position_for_four_consecutive_ticks():
    layout = CueLayout((.38, -.08), ((.40, -.08), (.57, .27)), (-.26, .01))
    view = CameraView.sample(np.random.default_rng(481))
    close = CameraArena(layout, 0, view, max_steps=12)
    distant = CameraArena(layout, 1, view, max_steps=12)
    for _ in range(SUCCESS_TICKS - 1):
        assert close.step((0., 0.))[1] == distant.step((0., 0.))[1] == 0
    assert close.step((0., 0.))[1] == 1
    assert distant.step((0., 0.))[1] == 0
    assert close.contacts == [{'tick': SUCCESS_TICKS, 'relation': 0, 'reward': 1}]
    assert distant.contacts == [{'tick': SUCCESS_TICKS, 'relation': 0, 'reward': 0}]


def test_motor_identifies_delay_from_issued_commands_and_displacements():
    effects = np.array(((1.5, 3., -.6), (4., -.9, -2.4)))
    zeros = np.zeros(3)
    command_history = [zeros, zeros, zeros]
    learner = CameraMotorLearner(3)
    learner.begin_trial(3609)
    for _ in range(CALIBRATION_TICKS):
        command_history.append(learner.act((20., -5.)))
        learner.observe(effects @ command_history[-4])
    assert learner.model.delay == 3
    assert learner.probe_ticks == list(range(1, CALIBRATION_TICKS + 1))
    np.testing.assert_allclose(learner.model.effects(), effects, atol=.18)


def test_source_reward_masking_and_fixed_motor_controls_share_starts_and_probes_are_charged():
    result = run([610011], courses=('both', 'unchanged'))
    stream = result['streams'][0]
    transfer = stream['transfer']['target']
    assert transfer['reset']['actions_sha256'] == transfer['source_reward_masked']['actions_sha256']
    assert transfer['carried']['first_choice'] == transfer['carried']['paying']
    assert transfer['opposite']['first_choice'] != transfer['opposite']['paying']
    assert transfer['carried']['replay']['layout'] == transfer['reset']['replay']['layout']
    assert transfer['carried']['replay']['actions'][:CALIBRATION_TICKS] == transfer['reset']['replay']['actions'][:CALIBRATION_TICKS]
    comparison = stream['motor']['courses']['both']
    assert comparison['fixed'][0]['probes'] == 0
    assert comparison['fixed'][0]['inferred_delay'] == stream['motor']['warmup_model']['delay']
    assert comparison['fresh'][0]['actions_sha256'] == comparison['fresh_at_boundary'][0]['actions_sha256']
    for condition in MOTOR:
        for reach in comparison[condition]:
            assert 0 <= reach['probes'] <= reach['commands'] <= result['max_steps']
            assert reach['success'] == any(hold['reward'] for hold in reach['contacts'])
