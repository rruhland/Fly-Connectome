"""Verify the TT18 variants alter geometry and constraints without breaking point rules."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip('mujoco')
from mujoco.rendering.classic.renderer import GLContext
from scripts.experimental.tt4_arena import BALL_MASS, TABLE_HALF_LENGTH
from scripts.experimental.tt8_service_arena import ServiceScore, TT8ServiceArena
from scripts.experimental.tt18_plant import PLANTS, PlantSettings, TT18PlantArena
from scripts.experimental.tt18_plant_audit import drop, draw_camera, flight, motion


def test_fixed_shape_heavy_ball_keeps_absolute_air_forces_and_changes_flight():
    old = TT18PlantArena()
    heavy = TT18PlantArena(plant=PlantSettings(ball_mass_scale=1.34))
    try:
        assert heavy.model.body('ball').mass[0] == pytest.approx(BALL_MASS * 1.34)
        np.testing.assert_allclose(heavy.model.body('ball').inertia, old.model.body('ball').inertia * 1.34)
        same = flight(old)
        altered = flight(heavy)
        assert same['applied_wrench'] == altered['applied_wrench']
        assert altered['velocity'][1] < same['velocity'][1] - .05
        assert altered['spin'][0] > same['spin'][0]
    finally:
        old.close()
        heavy.close()


def test_mirroring_the_table_material_mirrors_rebound_and_seam_scores_once():
    variants = []
    for left, right in ((48., 15.), (15., 48.)):
        arena = TT18PlantArena(plant=PlantSettings(left_damping=left, right_damping=right), frameskip=1)
        try:
            result = [drop(arena, x) for x in (-.37, 0., .37)]
            assert all(row['bounces'] == 1 and row['reason'] is None for row in result)
            variants.append(result)
        finally:
            arena.close()
    first, second = variants
    assert first[2]['rebound'] > first[0]['rebound'] + .25
    assert first[0]['rebound'] == pytest.approx(second[2]['rebound'])
    assert first[2]['rebound'] == pytest.approx(second[0]['rebound'])


def test_smaller_table_changes_edge_and_serving_position_and_restricts_motor_travel():
    stock = TT18PlantArena(body='gantry', frameskip=1)
    compact = TT18PlantArena(body='gantry', frameskip=1, plant=PLANTS['compact'])
    try:
        assert drop(stock, .74)['bounces'] == 1
        assert drop(compact, .74)['bounces'] == 0
        for arena in (stock, compact):
            arena.reset(seed=10, server=1, render=False)
        assert stock.ball[1] + TABLE_HALF_LENGTH == pytest.approx(compact.ball[1] + compact.table_half_length)
        initial = stock.model.joint('learner_lateral').range
        changed = compact.model.joint('learner_lateral').range
        np.testing.assert_allclose(changed, initial * .86)
        for _ in range(85):
            compact.step(np.array([1., 0., 0., 0.]), render=False)
        assert compact.data.ctrl[compact.model.actuator('learner_lateral').id] == pytest.approx(changed[1])
    finally:
        stock.close()
        compact.close()


def test_split_table_keeps_toss_apex_and_scored_descending_serve():
    stock = TT8ServiceArena(body='gantry')
    split = TT18PlantArena(body='gantry', plant=PlantSettings(left_damping=24., right_damping=24.01))
    hold = (0., 0., 0., 0.)
    actions = [(1., -1., 1., -.465517), (.334553, -1., 1., 0.), (0., -1., .235294, 0.),
               (0., -1., 0., 0.), (0., -.105685, 0., 0.)]
    actions += [hold] * 6 + [(0., 1., 0., 0.)] * 5 + [hold] * 100
    try:
        for arena in (stock, split):
            arena.reset(seed=10, server=1, render=False)
            for action in actions:
                arena.step(action, render=False)
                if arena.terminated or arena.truncated:
                    break
            assert arena.service_contact and arena.legal_serve
            assert arena._toss_apex - arena._toss_height > .16
            assert [(e['kind'], e['side']) for e in arena.events
                    if e['kind'] in ('serve_strike', 'bounce')][:3] == [('serve_strike', 1), ('bounce', 1), ('bounce', -1)]
            assert ServiceScore().record(arena.point_result()) == (1, 0)
        assert split._toss_apex == stock._toss_apex
    finally:
        stock.close()
        split.close()


@pytest.mark.skipif(GLContext is None or GLContext.__module__ != 'mujoco.egl',
                    reason='run with MUJOCO_GL=egl to exercise mounted native rendering')
@pytest.mark.parametrize('mount,parent', [('wrist', 'learner_wrist'), ('proximal', 'learner_bridge')])
def test_housing_adds_mass_to_its_link_and_camera_can_render_during_motion(mount, parent):
    stock = TT18PlantArena(body='gantry', width=64, height=48)
    settings = PlantSettings(payload_mount=mount, payload_mass=.05)
    changed = TT18PlantArena(body='gantry', plant=settings, width=64, height=48)
    try:
        assert changed.model.body(parent).mass[0] - stock.model.body(parent).mass[0] == pytest.approx(.05)
        receipt = draw_camera(changed, [[1., 0., 1., 1.]] * 2)
        assert receipt['parent'] == parent
        assert receipt['housing_on_camera_body']
        assert receipt['shape'] == [48, 64, 3]
        assert receipt['displacement_m'] > .001
        assert receipt['changed_pixel_fraction'] > .01
        assert changed.render_cameras().shape == (2, 48, 64, 3)
    finally:
        changed.close()
        stock.close()


def test_redistributing_the_same_mass_changes_motion_and_local_moments():
    settings = PlantSettings()
    stock = TT18PlantArena(body='gantry', plant=settings)
    variants = []
    try:
        for link in ('learner_wrist', 'learner_bridge'):
            before = stock.model.body(link)
            scale = 1 + .05 / before.mass[0]
            arena = TT18PlantArena(body='gantry',
                                   plant=replace(settings, link_mass_scales={link.removeprefix('learner_'): scale}))
            variants.append(arena)
            after = arena.model.body(link)
            assert after.mass[0] - before.mass[0] == pytest.approx(.05)
            np.testing.assert_allclose(after.inertia, scale * before.inertia)
        tape = [[1., -1., 1., -1.]] * 8
        a, b = [motion(arena, tape)[0] for arena in variants]
        assert np.linalg.norm(a - b, axis=1).max() > .002
    finally:
        stock.close()
        for arena in variants:
            arena.close()
