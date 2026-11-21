"""Verify the camera boundary, physical strikes and repeatable point scoring."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from tt3_arena import (BALL_RADIUS, FOREARM, HZ, IMAGE_HEIGHT, IMAGE_WIDTH, JOINT_RATES,
                      NET_HEIGHT, TABLE_HALF_LENGTH, TT3Arena, UPPER_ARM, arm_pose,
                      heldout_cases)


HOLD = (0., 0., 0.)


def test_joint_rates_move_a_rigid_articulated_arm_in_three_dimensions():
    arena = TT3Arena(10)
    starting_joints = arena.joints
    starting_paddle = arena.paddle
    action = (1., -.7, .2)
    frames, reward, done, _ = arena.step(action)
    assert arena.joints == pytest.approx(np.array(starting_joints) + np.array(action) * JOINT_RATES / HZ)
    base, elbow, paddle, normal = arm_pose(arena.joints)
    assert math.dist(base, elbow) == pytest.approx(UPPER_ARM)
    assert math.dist(elbow, paddle) == pytest.approx(FOREARM)
    assert np.linalg.norm(normal) == pytest.approx(1.)
    assert all(before != after for before, after in zip(starting_paddle, paddle))
    assert len(frames) == 2
    assert all(image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3) and image.dtype == np.uint8
               for image in frames)
    assert reward == 0 and not done


def test_appearance_seed_and_distractor_have_no_effect_on_point_physics():
    plain = TT3Arena(7, view_seed=3)
    decoy = TT3Arena(7, view_seed=3, distractor=True)
    new_colors = TT3Arena(7, view_seed=4)
    for camera in (0, 1):
        assert not np.array_equal(plain.frame()[camera], decoy.frame()[camera])
        assert not np.array_equal(plain.frame()[camera], new_colors.frame()[camera])
    masks = [np.any(a != b, axis=-1) for a, b in zip(plain.frame(), decoy.frame())]
    while not plain.done:
        for arena in (plain, decoy, new_colors):
            arena.step(HOLD, render=False)
    assert plain.events == decoy.events == new_colors.events
    assert plain.point_result() == decoy.point_result() == new_colors.point_result()
    assert any(not np.array_equal(mask, np.any(a != b, axis=-1))
               for mask, a, b in zip(masks, plain.frame(), decoy.frame()))
    for a, b in zip(heldout_cases(4, 12), heldout_cases(4, 12, distractor=True)):
        assert (a.seed, a.view_seed) == (b.seed, b.view_seed)
        assert not a.distractor and b.distractor


def test_swept_paddle_collision_can_miss_by_height_or_depth_and_transmits_motion():
    def near_paddle(z_offset=0., y_offset=.065):
        arena = TT3Arena(8)
        x, y, z = arena.paddle
        arena.ball = (x, y + y_offset, z + z_offset)
        arena.velocity = (0., -4., 0.)
        arena._bounced = True
        return arena

    still = near_paddle()
    still.step(HOLD, render=False)
    assert still.contacts == 1
    assert still.velocity[1] > 0.
    moving = near_paddle()
    moving.step((0., .55, -1.), render=False)
    assert moving.contacts == 1
    assert moving.velocity[1] != pytest.approx(still.velocity[1], abs=.03)
    above = near_paddle(z_offset=.26)
    behind = near_paddle(y_offset=-.14)
    for _ in range(3):
        above.step(HOLD, render=False)
        behind.step(HOLD, render=False)
    assert above.contacts == behind.contacts == 0
    assert above.velocity[1] < 0. and behind.velocity[1] < 0.


def test_public_reward_requires_clearing_net_and_landing_on_opposite_half():
    arena = TT3Arena()
    arena._hitter = 1
    arena.ball = (0., -.012, NET_HEIGHT - BALL_RADIUS)
    arena.velocity = (0., 3., 0.)
    _, reward, done, info = arena.step(HOLD, render=False)
    assert reward == -1 and done
    assert info['legal_returns'] == 0 and info['points_against'] == 1
    assert arena.point_result()['reason'] == 'net'
    arena.reset(render=False)
    arena._hitter = 1
    arena.ball = (0., TABLE_HALF_LENGTH / 2, BALL_RADIUS + .007)
    arena.velocity = (0., .4, -1.7)
    _, reward, done, info = arena.step(HOLD, render=False)
    assert reward == 0 and not done
    assert info['legal_returns'] == 1 and info['points_for'] == 0
    while not arena.done:
        arena.step(HOLD, render=False)
    assert arena.point_result()['points_for'] == 1
    assert arena.point_result()['reason'] == 'second_bounce'


def test_height_course_brings_legal_openings_to_distinct_heights_at_the_same_depth():
    heights = []
    for case in heldout_cases(710, 24, variant='height'):
        arena = TT3Arena(case.seed, view_seed=case.view_seed, variant=case.variant)
        while arena.ball[1] > -1.27 and not arena.done:
            arena.step(HOLD, render=False)
        assert not arena.done
        assert [event['side'] for event in arena.events if event['kind'] == 'bounce'] == [1]
        heights.append(arena.ball[2])
    assert sum(height < .23 for height in heights) >= 8
    assert sum(height > .47 for height in heights) >= 8


def test_two_scripted_paddle_positions_play_through_an_opponent_return():
    arena = TT3Arena(6613836788881966817)
    first = np.array((.700872, -.599573, 1.508387))
    second = np.array((-.280875, -1.012256, 1.913319))
    goal = first
    while not arena.done:
        command = np.clip((goal - arena.joints) * HZ / JOINT_RATES, -1., 1.)
        _, _, _, info = arena.step(command, render=False)
        if any(event['kind'] == 'contact' and event['side'] == -1 for event in info['events']):
            goal = second
    result = arena.point_result()
    assert result['contacts'] == result['legal_returns'] == 2
    assert result['rival_returns'] == 1
    assert result['points_for'] == 1
    hit_sides = [event['side'] for event in arena.events if event['kind'] == 'contact']
    assert hit_sides[:3] == [1, -1, 1]
