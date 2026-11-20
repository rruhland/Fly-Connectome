import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from body_reaching_arena import (BODY_NAMES, BODY_SPECS, CAMERA_CENTER, CAMERA_SIZE, GOAL_TOLERANCE,
                                 MIN_START_DISTANCE, SUCCESS_TICKS, TASK_X, TASK_Y, TICK_SECONDS,
                                 ReachCase, ReachingArena, _arm_configuration, decode_markers,
                                 render_markers, sample_case)


def test_observation_decodes_only_the_two_rasterized_markers():
    position = (.487, -.148)
    frame = render_markers(position, position)
    assert frame.shape == (2, CAMERA_SIZE, CAMERA_SIZE)
    assert frame.dtype == bool
    assert np.sum(frame, axis=(1, 2)).tolist() == [9, 16]
    observed_tip, observed_goal = decode_markers(frame)
    np.testing.assert_array_equal(observed_tip, observed_goal)
    np.testing.assert_allclose(observed_tip, position, atol=.5 / CAMERA_CENTER)
    assert not np.array_equal(observed_tip, position)

    arena = ReachingArena('arm', ReachCase(start=(.50, .20), target=(.26, -.20)))
    arena.frame = lambda: frame
    np.testing.assert_array_equal(arena.observation(), (observed_tip, observed_goal))


def test_carriage_translation_and_arm_actuation_are_structurally_different():
    deltas = {}
    for body in ('carriage', 'arm'):
        deltas[body] = []
        for y in (-.30, .30):
            case = ReachCase(start=(.40, y), target=(.55, -y))
            arena = ReachingArena(body, case)
            arena.step((1., 0.))
            deltas[body].append(np.subtract(arena.ground_truth_position, case.start))

    horizontal_motion = 1.15 * .80 * TICK_SECONDS
    np.testing.assert_allclose(deltas['carriage'], [(horizontal_motion, 0.)] * 2)
    assert deltas['arm'][0][0] > 0 > deltas['arm'][1][0]
    assert all(delta[1] > 0 for delta in deltas['arm'])
    for y, delta in zip((-.30, .30), deltas['arm']):
        assert np.linalg.norm(np.array([.40, y]) + delta) == pytest.approx(.50)


def test_remapped_arm_reproduces_variant_motion_with_different_bounded_commands():
    case = ReachCase(start=(.40, -.31), target=(.45, .20))
    variant = ReachingArena('arm_variant', case)
    remapped = ReachingArena('arm_remapped', case)
    nominal = ReachingArena('arm', case)
    for a, b in ((1., 0.), (0., 0.), (1.4, -.3), (-.1, .9), (0., 0.)):
        shifted_observation = remapped.step((a, b))
        variant_observation = variant.step((-b, a))
        nominal.step((a, b))
        np.testing.assert_allclose(remapped.ground_truth_position, variant.ground_truth_position)
        np.testing.assert_array_equal(shifted_observation, variant_observation)
    assert math.dist(nominal.ground_truth_position, remapped.ground_truth_position) > .025


@pytest.mark.parametrize('body', BODY_NAMES)
def test_matched_sampled_and_boundary_cases_are_physically_reachable(body):
    rng = np.random.default_rng(8226)
    sampled_cases = [sample_case(rng) for _ in range(12)]
    assert all(math.dist(case.start, case.target) >= MIN_START_DISTANCE for case in sampled_cases)
    corners = [(x, y) for x in TASK_X for y in TASK_Y]
    cases = [*sampled_cases, *(ReachCase(start, target) for start, target in zip(corners, corners[::-1]))]
    spec = BODY_SPECS[body]
    drive = np.array(spec.drive)
    for case in cases:
        arena = ReachingArena(body, case)
        np.testing.assert_allclose(arena.ground_truth_position, case.start)
        np.testing.assert_allclose(arena.observation(), (case.start, case.target), atol=.5 / CAMERA_CENTER)
        goal = case.target if spec.links is None else _arm_configuration(case.target, spec.links)
        while not arena.done:
            # A test-only servo with privileged geometry checks task feasibility.
            requested_velocity = 6 * np.subtract(goal, arena._configuration)
            command = np.linalg.solve(drive, requested_velocity)
            arena.step(command / max(1., np.linalg.norm(command, ord=np.inf)))
        assert arena.success


@pytest.mark.parametrize('start,target,expected', [(.400, .454, True), (.411, .467, False)])
def test_success_uses_true_tolerance_for_the_full_hold(start, target, expected):
    arena = ReachingArena('carriage', ReachCase((start, 0.), (target, 0.)), max_steps=SUCCESS_TICKS)
    visible_start, visible_target = arena.observation()
    assert (math.dist(visible_start, visible_target) > GOAL_TOLERANCE) is expected
    for _ in range(SUCCESS_TICKS):
        assert not arena.done
        arena.step((0., 0.))
    assert arena.done and arena.success is expected
    assert arena.normalized_error == 1
    assert arena.weighted_normalized_error == pytest.approx(1)
    with pytest.raises(RuntimeError, match='episode has ended'):
        arena.step((0., 0.))


def test_weighted_error_uses_full_matched_horizon_and_render_is_separate():
    arena = ReachingArena('carriage', ReachCase((.28, -.11), (.44, -.11)), max_steps=12)
    for action in [(1., 0.)] * 4 + [(0., 0.)] * 2:
        arena.step(action)
    assert arena.success and arena.steps == 6
    errors = [math.dist(point, arena.case.target) for point in arena.trace]
    errors += [errors[-1]] * (arena.max_steps - arena.steps)
    assert arena.normalized_error == pytest.approx(errors[-1] / errors[0])
    expected_score = np.average(errors[1:], weights=np.linspace(1, 2, arena.max_steps)) / errors[0]
    assert arena.weighted_normalized_error == pytest.approx(expected_score)
    display = arena.render_rgb(scale=2)
    assert display.shape == (2 * CAMERA_SIZE, 2 * CAMERA_SIZE, 3)
    assert display.dtype == np.uint8
    assert np.any(np.all(display == (255, 191, 116), axis=-1))
    assert np.any(np.all(display == (95, 225, 170), axis=-1))
