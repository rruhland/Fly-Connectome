import numpy as np
import pytest

pytest.importorskip('mujoco')

from scripts.experimental.tt4_arena import mujoco
from scripts.experimental.tt19_so101 import WRIST_CAMERA
from scripts.experimental.tt20_fixture import CLAMP_FORWARD, CLAMP_RISE, TT20Arena


@pytest.mark.parametrize('hardware', ('clamp', 'camera'))
@pytest.mark.parametrize('landed', (False, True))
def test_nonplaying_hardware_contact_is_dead_and_does_not_score_a_bounce(hardware, landed):
    arena = TT20Arena()
    try:
        arena.reset(seed=7, server=-1, render=False)
        arena._received_bounce = landed
        if hardware == 'clamp':
            support = arena.data.geom('tt20_clamp_0_beam')
            arena.ball[:] = support.xpos + (0., 0., .034)
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
        assert fixture['collider'].startswith('tt20_clamp_' if hardware == 'clamp' else 'tt20_rgb_')
        assert point['kind'] == 'point'
        assert arena.contacts == arena.legal_returns == arena.rival_contacts == arena.rival_returns == 0
    finally:
        arena.close()


def test_moved_end_clamp_sits_below_table_surface():
    arena = TT20Arena()
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
    arena = TT20Arena(view_mode='wrist')
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
    standard = TT20Arena(view_mode='wrist')
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
