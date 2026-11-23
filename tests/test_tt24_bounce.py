"""Check the public sensor boundary and distinguish actual rebound arcs from contacts."""

import numpy as np
import pytest

pytest.importorskip('mujoco')
from mujoco.rendering.classic.renderer import GLContext
from scripts.experimental.tt19_so101 import CAMERA_BODY
from scripts.experimental.tt24_bounce import BALL_RADIUS, BounceArena, MIN_AIRBORNE, MIN_RISE


@pytest.fixture(scope='module')
def arena():
    result = BounceArena(view='none')
    yield result
    result.close()


def _finish(arena):
    rewards = []
    for _ in range(arena.max_steps):
        _, reward, done = arena.step(np.zeros(arena.action_size))
        rewards.append(reward)
        if done:
            return arena.judge_metrics(), rewards
    pytest.fail('episode did not terminate at the configured horizon')


def test_verified_rebounds_are_separate_upward_arcs(arena):
    arena.reset(seed=42)
    result, rewards = _finish(arena)
    events = result['rebound_events']
    assert len(events) == result['rebounds'] == sum(rewards) >= 2
    assert result['longest_consecutive_rebounds'] == len(events)
    for event in events:
        assert event['contact_time'] < event['launch_time'] < event['apex_time']
        assert event['incoming_vz'] < 0 < event['launch_vz']
        assert event['rise_m'] >= MIN_RISE
        assert event['airborne_s'] >= MIN_AIRBORNE
    assert all(prior['apex_time'] < current['contact_time'] for prior, current in zip(events, events[1:]))


@pytest.mark.parametrize('collider,position', (
    ('table', (.22, -.52, .84)),
    ('floor', (.22, -1.42, .019)),
    ('learner_upper_arm_ball_contact_1', None),
))
def test_contacts_from_other_surfaces_fail_without_reward(arena, collider, position):
    arena.reset(seed=42)
    arena._ball_position[:] = arena._data.geom(collider).xpos if position is None else position
    arena._ball_velocity[:] = 0.
    result, rewards = _finish(arena)
    assert result['termination'] == 'non_paddle_contact'
    assert result['other_collider'] == collider
    assert sum(rewards) == result['rebounds'] == 0


def test_low_height_paddle_chatter_does_not_score(arena):
    arena.reset(seed=42)
    arena._ball_position[:] = arena._data.geom_xpos[arena._blade] + (0., 0., BALL_RADIUS + .014)
    arena._ball_velocity[:] = 0.
    arena._ball_spin[:] = 0.
    result, rewards = _finish(arena)
    assert result['paddle_contacts'] > 2
    assert sum(rewards) == result['rebounds'] == result['longest_consecutive_rebounds'] == 0


def test_changed_physics_and_reward_have_independent_controls(arena):
    baseline = arena.reset(seed=42, split='held', regime='standard')
    baseline_result, _ = _finish(arena)
    arena.reset(seed=42, split='held', regime='changed')
    changed_result, _ = _finish(arena)
    assert changed_result['drop']['drag'] == pytest.approx(baseline_result['drop']['drag'] * 1.25)
    assert changed_result['drop']['damping'] > baseline_result['drop']['damping']
    assert changed_result['rebound_events'] != baseline_result['rebound_events']
    restored = arena.reset(seed=42, split='held', regime='standard')
    np.testing.assert_array_equal(restored['joints'], baseline['joints'])
    restored_result, _ = _finish(arena)
    assert restored_result == baseline_result

    graded = BounceArena(view='none', apex_reward_weight=.25)
    graded.reset(seed=42, split='held', regime='standard')
    graded_result, rewards = _finish(graded)
    assert graded_result['rebound_events'] == baseline_result['rebound_events']
    expected_bonus = sum(.25 * min(event['rise_m'] / .12, 1.) for event in graded_result['rebound_events'])
    assert sum(rewards) == pytest.approx(baseline_result['rebounds'] + expected_bonus)


def test_alternate_physical_arm_and_public_action_space(arena):
    alternate = BounceArena(view='none', proprioception=False, arm='alternate')
    assert alternate.action_axes == arena.action_axes
    assert alternate.reset(seed=42) == {}
    observation, reward, done = alternate.step(np.zeros(arena.action_size))
    assert observation == {} and reward == 0. and not done
    common_joint = 'learner_elbow_flex'
    assert alternate._model.body('learner_elbow').pos[0] < arena._model.body('learner_elbow').pos[0]
    assert alternate._model.joint(common_joint).axis[0] > arena._model.joint(common_joint).axis[0]
    assert alternate._model.body('tt24_wrist_payload').mass[0] == pytest.approx(.042)
    assert not np.array_equal(alternate._model.body(CAMERA_BODY).pos, arena._model.body(CAMERA_BODY).pos)


@pytest.mark.skipif(GLContext is None or GLContext.__module__ != 'mujoco.egl',
                    reason='run with MUJOCO_GL=egl to test native wrist RGB')
def test_wrist_rgb_and_optional_joint_sensors():
    arena = BounceArena(width=96, height=72)
    try:
        rgb = arena.reset(seed=42)
        assert set(rgb) == {'rgb', 'joints', 'joint_velocity'}
        assert rgb['rgb'].shape == (72, 96, 3) and rgb['rgb'].dtype == np.uint8
        red, green, blue = np.moveaxis(rgb['rgb'], -1, 0)
        orange = (red > 68) & (green > .20 * red) & (green < .68 * red) & (blue < .34 * red)
        assert np.count_nonzero(orange) > 8
        observation, _, _ = arena.step(np.zeros(arena.action_size), render=False)
        assert set(observation) == {'joints', 'joint_velocity'}
        for values in observation.values():
            assert values.shape == (arena.action_size,)
    finally:
        arena.close()
