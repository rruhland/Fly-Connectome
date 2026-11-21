"""Verify the receiving simulator's visible interface and physical endpoints."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


if importlib.util.find_spec('mujoco') is None:
    pytest.skip('table-tennis optional dependencies are required', allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experimental.tt4_arena import (BALL_MASS, BALL_RADIUS, NET_HEIGHT, TABLE_HEIGHT,
                                            TT4Arena)
from scripts.experimental.tt4_arena_audit import measure_controls, measure_net, measure_table_drop


def test_stereo_rgb_replays_with_only_four_action_channels():
    arena = TT4Arena(12, view_seed=32, distractor=True, baseline=.064)
    opening = arena.reset(seed=12)
    assert opening.shape == (2, 72, 96, 3)
    assert opening.dtype == np.uint8
    assert not np.array_equal(opening[0], opening[1])
    assert arena.cameras[1].eye[0] - arena.cameras[0].eye[0] == pytest.approx(.064)
    assert arena.model.vis.map.znear * arena.model.stat.extent < .10
    action = np.array((.88, -.35, .36, -.56))
    initial_joints = arena.joints
    first = arena.step(action)
    np.testing.assert_array_equal(arena.reset(seed=12), opening)
    rerun = arena.step(action)
    np.testing.assert_array_equal(rerun[0], first[0])
    assert rerun[1:] == first[1:]
    assert np.linalg.norm(arena.joints - initial_joints) > .01
    with pytest.raises(ValueError, match='four finite'):
        arena.step(np.zeros(3))
    with pytest.raises(ValueError, match='four finite'):
        arena.step((0., 1., 0., np.nan))


def test_regulation_geometry_and_real_table_and_net_contacts():
    arena = TT4Arena()
    model = arena.model
    assert model.geom('ball').size[0] == pytest.approx(BALL_RADIUS)
    assert model.body('ball').mass[0] == pytest.approx(BALL_MASS)
    table = model.geom('table')
    assert table.size[:2] * 2 == pytest.approx((1.525, 2.74))
    assert table.pos[2] + table.size[2] == pytest.approx(TABLE_HEIGHT)
    net = model.geom('net')
    assert net.pos[2] + net.size[2] == pytest.approx(TABLE_HEIGHT + NET_HEIGHT)
    drop = measure_table_drop(arena, .30)
    assert drop['rebound_m'] == pytest.approx(.23, abs=.015)
    assert .83 < drop['restitution'] < .97
    assert 0. < drop['contact_s'] < .015
    strike = measure_net(arena)
    assert strike['location'][1] == pytest.approx(0., abs=.013)
    assert strike['location'][2] < TABLE_HEIGHT + NET_HEIGHT
    assert abs(strike['exit_velocity'][1]) < abs(strike['incoming'][1]) / 2


def test_spin_changes_free_flight():
    arena = TT4Arena()
    dof = arena.model.jnt_dofadr[arena.model.joint('ball_free').id]
    heights = []
    for spin in (-150., 150.):
        arena.reset(seed=4, render=False)
        arena.ball[:] = (0., 1.40, 1.77)
        arena.velocity[:] = (0., -4., 0.)
        arena.data.qvel[dof + 3:dof + 6] = (spin, 0., 0.)
        for _ in range(9):
            arena.step(np.zeros(4), render=False)
        heights.append(arena.ball[2])
    assert heights[0] - heights[1] > .01


def test_oracle_contacts_land_legally_and_a_physical_rival_returns():
    arena = TT4Arena(max_steps=165)
    controls = measure_controls(arena, (0, 1, 2), random_seed=85416)
    stationary, random, oracle = (controls[key] for key in ('stationary', 'random', 'oracle'))
    assert stationary['legal_two_bounce_openings'] == random['legal_two_bounce_openings'] == 3
    assert stationary['legal_returns'] == random['legal_returns'] == 0
    assert oracle['opening_return_points'] == 3
    assert oracle['repeated_return_points'] >= 1
    assert oracle['rival_returns'] >= 2
    assert oracle['median_player_racket_exit_vy_mps'] > 2.
