"""Verify the stock-scale SO-101 mechanism and the scored fixture's interfaces."""

import pytest

pytest.importorskip('mujoco')
from mujoco.rendering.classic.renderer import GLContext
from scripts.experimental.tt19_so101 import AXES, SO101Arena
from scripts.experimental.tt19_so101_audit import (camera_probe, check_grasp, check_joint_travel,
                                                   motion_parity, scoring_witnesses, upstream_match)


@pytest.fixture(scope='module')
def preview():
    arena = SO101Arena(detail='preview', venue='hall', width=192, height=144, wrist_near=.012)
    yield arena
    arena.close()


def test_upstream_kinematics_inertia_and_joint_targets(preview):
    assert preview.action_size == len(AXES) == 6
    assert 1 / preview.control_dt == 30.
    source = upstream_match(preview)
    assert len(source['reference_bodies']) == 9
    assert len(source['joints']) == preview.action_size
    drive = check_joint_travel(preview)
    assert drive['active_targets'][0] == drive['limits'][0][1]
    assert drive['active_targets'][-1] == drive['limits'][-1][0]


def test_actual_jaw_can_hold_and_release_the_paddle(preview):
    grasp = check_grasp(preview)
    assert grasp['open_angle_rad'] > .44
    assert grasp['paddle_height_before_fall_m'] - grasp['paddle_height_after_fall_m'] > .3


def test_default_station_supports_a_legal_serve_and_winning_legal_return(preview):
    examples = scoring_witnesses(preview)
    assert examples['serve_point']['evaluator']['legal_serve']
    assert examples['return_point']['evaluator']['legal_returns'] == 1
    assert examples['return_point']['evaluator']['points_for'] == 1


def test_reduced_meshes_leave_the_physical_replay_bit_exact(preview):
    full = SO101Arena(venue='hall')
    try:
        assert len(motion_parity(full, preview)) == 8
    finally:
        full.close()


@pytest.mark.skipif(GLContext is None or GLContext.__module__ != 'mujoco.egl',
                    reason='run with MUJOCO_GL=egl to test native wrist RGB')
def test_wrist_rgb_resolves_and_follows_gripper_and_blade(preview, tmp_path):
    camera = camera_probe(preview, tmp_path)
    assert camera['visible_pixels']['blade'] > 200
    assert camera['visible_pixels']['moving_jaw'] > 80
    assert camera['center_motion_m'] > .03
