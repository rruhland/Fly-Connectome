import numpy as np
import pytest

pytest.importorskip('mujoco')

from scripts.experimental.tt4_arena import mujoco
from scripts.experimental.tt19_so101 import GRIP_BODY, WRIST_CAMERA
from scripts.experimental.tt20_fixture import ARCHIVED_FIXTURE, CLAMP_FORWARD, CLAMP_RISE, TT20Arena
from scripts.experimental.tt24_bounce import BounceArena
from scripts.experimental.tt25_reference_receiver import ReferenceReceiver
from scripts.experimental.tt29_arm_clearance import ArmClearance


@pytest.mark.parametrize('hardware', ('clamp', 'camera'))
@pytest.mark.parametrize('landed', (False, True))
def test_nonplaying_hardware_contact_is_dead_and_does_not_score_a_bounce(hardware, landed):
    arena = TT20Arena()
    try:
        arena.reset(seed=7, server=-1, render=False)
        arena._received_bounce = landed
        if hardware == 'clamp':
            support = arena.data.geom('tt29_clamp_0_upper_jaw')
            arena.ball[:] = support.xpos + (0., -.010, .022)
            arena.velocity[:] = (0., 0., -.3)
        else:
            camera = arena.data.camera(WRIST_CAMERA)
            forward = -camera.xmat.reshape(3, 3)[:, 2]
            arena.ball[:] = camera.xpos + .015 * forward
            arena.velocity[:] = -.3 * forward
        mujoco.mj_forward(arena.model, arena.data)
        _, reward, done, capped, _ = arena.step(np.zeros(arena.action_size), render=False)
        assert done and not capped and reward == (-1 if landed else 1)
        assert arena.reason == 'fixture_contact'
        fixture, point = arena.events
        assert fixture['kind'] == 'fixture_contact'
        assert fixture['collider'].startswith('tt29_clamp_' if hardware == 'clamp' else 'tt29_camera_')
        assert point['kind'] == 'point'
        assert arena.contacts == arena.legal_returns == arena.rival_contacts == arena.rival_returns == 0
    finally:
        arena.close()


def test_moved_end_clamp_sits_below_table_surface():
    arena = TT20Arena(fixture=ARCHIVED_FIXTURE)
    pedestal = TT20Arena(mount='pedestal', wrist_optics='legacy')
    try:
        base = arena.data.body('learner_base').xpos
        np.testing.assert_allclose(base - pedestal.data.body('learner_base').xpos, (0., CLAMP_FORWARD, CLAMP_RISE))
        flange = arena.data.geom('tt20_clamp_flange')
        np.testing.assert_allclose(flange.xpos + (0., 0., arena.model.geom(flange.id).size[2]), base)
        table = arena.model.geom('table')
        playing_height = table.pos[2] + table.size[2]
        clamps = [index for index in range(arena.model.ngeom) if arena.model.geom(index).name.startswith('tt20_clamp_')]
        assert all(arena.data.geom_xpos[index, 2] < playing_height for index in clamps)
        assert arena.model.vis.map.znear * arena.model.stat.extent == pytest.approx(.003)
    finally:
        arena.close()
        pedestal.close()


@pytest.mark.parametrize('arena_type', (ReferenceReceiver, BounceArena))
def test_table_edge_fixture_has_fixed_base_and_opposing_jaws(arena_type):
    arena = arena_type()
    try:
        if isinstance(arena, BounceArena):
            model, data = arena._model, arena._data
        else:
            model, data = arena.model, arena.data
        assert arena.fixture == 'table-edge'
        base = model.body('learner_base')
        assert base.parentid[0] == base.jntnum[0] == 0
        assert model.body('tt20_table_end_clamp').jntnum[0] == 0
        table = model.geom('table')
        near_edge = -table.size[1]
        table_top = table.pos[2] + table.size[2]
        underside = table.pos[2] - table.size[2]
        plate = model.geom('tt29_clamp_top_plate')
        assert plate.pos[2] - plate.size[2] == pytest.approx(table_top)
        assert base.pos[2] - .0024 == pytest.approx(plate.pos[2] + plate.size[2])
        assert base.pos[1] - near_edge == pytest.approx(.026)
        for index in range(2):
            prefix = f'tt29_clamp_{index}'
            upper, lower, shoe = (model.geom(prefix + suffix) for suffix in ('_upper_jaw', '_lower_jaw', '_shoe'))
            assert upper.pos[2] - upper.size[2] == pytest.approx(table_top)
            assert lower.pos[2] + lower.size[2] < underside
            assert shoe.pos[2] + shoe.size[2] == pytest.approx(underside)
            assert shoe.pos[1] - shoe.size[1] > near_edge
            assert data.geom(prefix + '_handle').xpos[2] < lower.pos[2]
    finally:
        arena.close()


def test_board_lens_is_rigid_and_reaches_the_court_at_ready_pose():
    arena = ReferenceReceiver(view_mode='wrist+fixed')
    try:
        arena.reset(seed=42, render=False)
        handed = arena.data.body(GRIP_BODY)
        camera = arena.camera_calibration()['wrist']
        grip_pose = handed.xmat.reshape(3, 3).copy()
        offset = grip_pose.T @ (camera['position'] - handed.xpos)
        rotation = grip_pose.T @ camera['rotation']
        np.testing.assert_allclose(offset, (.006, .054, -.035), atol=1e-9)
        assert np.dot(-camera['rotation'][:, 2], (0., 1., 0.)) > .95
        hit = np.empty(1, dtype=np.int32)
        distance = mujoco.mj_ray(arena.model, arena.data, camera['position'],
                                 -np.ascontiguousarray(camera['rotation'][:, 2]), None, True, -1, hit)
        assert distance > .2
        joint = arena.model.joint('learner_wrist_roll')
        arena.data.qpos[joint.qposadr[0]] += .35
        moved = arena.camera_calibration()['wrist']
        next_grip = handed.xmat.reshape(3, 3)
        np.testing.assert_allclose(next_grip.T @ (moved['position'] - handed.xpos), offset, atol=1e-9)
        np.testing.assert_allclose(next_grip.T @ moved['rotation'], rotation, atol=1e-9)
        assert np.linalg.norm(moved['position'] - camera['position']) > .015
    finally:
        arena.close()


def test_camera_foot_and_bracket_clear_the_travel_of_the_moving_jaw():
    arena = ReferenceReceiver(detail='cad')
    try:
        model, data = arena.model, arena.data
        jaw = model.body('learner_moving_jaw_so101_v1').id
        moving = next(index for index, body in enumerate(model.geom_bodyid) if body == jaw)
        hardware = [model.geom(name).id for name in ('tt29_camera_foot', 'tt29_camera_bracket')]
        hinge = model.joint('learner_gripper')
        for q in np.linspace(*hinge.range, 31):
            data.qpos[hinge.qposadr[0]] = q
            mujoco.mj_kinematics(model, data)
            for geom in hardware:
                assert mujoco.mj_geomDistance(model, data, geom, moving, .030, None) > 0.
    finally:
        arena.close()


def test_side_on_handle_fits_both_jaws_and_holds_the_lower_pose():
    arena = ReferenceReceiver(detail='cad', paddle_grip='side-on')
    try:
        arena.reset(seed=42, render=False)
        model, data = arena.model, arena.data
        clearance = ArmClearance(model)
        gripper = model.body(GRIP_BODY).id
        moving_jaw = model.body('learner_moving_jaw_so101_v1').id
        moving = next(index for index, body in enumerate(model.geom_bodyid) if body == moving_jaw)
        fixed = next(index for index, body in enumerate(model.geom_bodyid)
                     if body == gripper and model.mesh(model.geom_dataid[index]).name == 'wrist_roll_follower_so101_v1')
        for tick in range(21):
            assert arena.grasped
            assert not clearance.violations(data)
            if tick in (0, 20):
                paddle = data.body('learner_paddle')
                grip_rotation = data.xmat[gripper].reshape(3, 3)
                handle_axis = paddle.xmat.reshape(3, 3)[:, 2]
                assert -grip_rotation[:, 1] @ handle_axis > np.cos(np.deg2rad(1.))
                finger_axis = np.ascontiguousarray(grip_rotation[:, 0])
                for jaw, ray in ((fixed, -finger_axis), (moving, finger_axis)):
                    contact = mujoco.mj_rayMesh(model, data, jaw, paddle.xpos, ray)
                    assert contact == pytest.approx(.0085, abs=.0001)
                assert data.geom('learner_racket').xpos[2] == pytest.approx(1.040, abs=.005)
            if tick < 20:
                arena.step(np.zeros(arena.action_size), render=False)
    finally:
        arena.close()


def test_arm_clearance_rejects_a_pose_that_intersects_nonadjacent_links():
    arena = ReferenceReceiver(detail='preview')
    try:
        guard = ArmClearance(arena.model)
        assert not guard.violations(arena.data)
        arena.data.qpos[arena._arm_ids[1][0][:5]] = (-.097, 1.745, 1.093, 1.584, -.642)
        assert any('learner_shoulder' in first and 'learner_wrist' in second and value < -.030
                   for first, second, value in guard.violations(arena.data))
    finally:
        arena.close()


def test_marking_ablation_preserves_role_palette():
    known = TT20Arena(marked=True)
    natural = TT20Arena(marked=False)
    try:
        known._color_roles()
        natural._color_roles()
        for name in ('ball', 'learner_racket', 'rival_racket'):
            np.testing.assert_array_equal(known.model.geom(name).rgba, natural.model.geom(name).rgba)
        assert any(known.model.geom(i).name.startswith('tt18_landmark_') for i in range(known.model.ngeom))
        assert not any(natural.model.geom(i).name.startswith(('tt18_landmark_', 'tt19_rim_', 'tt19_tip_mark'))
                       for i in range(natural.model.ngeom))
    finally:
        natural.close()
        known.close()


def test_outboard_aperture_has_no_opaque_neutral_center_hit_before_near_plane():
    arena = TT20Arena(view_mode='wrist', fixture=ARCHIVED_FIXTURE)
    try:
        camera = arena.data.camera(WRIST_CAMERA)
        forward = -camera.xmat.reshape(3, 3)[:, 2]
        geom = np.empty(1, dtype=np.int32)
        depth = mujoco.mj_ray(arena.model, arena.data, camera.xpos, forward, None, True, -1, geom)
        assert depth > arena.model.vis.map.znear * arena.model.stat.extent
        assert arena.model.geom(geom[0]).name in ('learner_racket', 'table', 'net')
    finally:
        arena.close()


def test_ultrawide_wrist_keeps_the_aperture_and_raises_the_aim():
    standard = TT20Arena(view_mode='wrist', fixture=ARCHIVED_FIXTURE)
    wide = TT20Arena(view_mode='wrist', wrist_optics='ultrawide')
    try:
        original = standard.data.camera(WRIST_CAMERA)
        altered = wide.data.camera(WRIST_CAMERA)
        np.testing.assert_array_equal(original.xpos, altered.xpos)
        base, pitched = original.xmat.reshape(3, 3), altered.xmat.reshape(3, 3)
        np.testing.assert_allclose(base[:, 0], pitched[:, 0], atol=1e-12)
        assert np.rad2deg(np.arccos(base[:, 2] @ pitched[:, 2])) == pytest.approx(20.)
        assert wide.lens_intrinsics()['wrist']['height'] == 720
        assert wide.lens_intrinsics()['wrist']['fovy'] == 126.
    finally:
        wide.close()
        standard.close()


@pytest.mark.parametrize(('mode', 'stream'), (('wrist', 'wrist'), ('outside', 'left'), ('right', 'right')))
def test_single_view_exposes_only_the_selected_rgb_stream(mode, stream):
    arena = TT20Arena(view_mode=mode, width=96, height=72)
    try:
        frames = arena.reset(seed=10, server=1)
        assert list(frames) == [stream]
        assert frames[stream].shape == (72, 96, 3)
        assert frames[stream].dtype == np.uint8
        lens = arena.lens_intrinsics()[stream]
        assert lens['width'] == 96 and lens['height'] == 72
    finally:
        arena.close()
