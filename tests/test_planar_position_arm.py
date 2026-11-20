import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from body_reaching_arena import (CAMERA_CENTER, CAMERA_SIZE, GOAL_TOLERANCE, SUCCESS_TICKS, TASK_X,
                                 TASK_Y, TICK_SECONDS, ReachCase, decode_markers, sample_case)
from planar_position_arm import (JOINT_LIMITS, LINK_LENGTHS, MAX_JOINT_SPEED, MAX_RELATIVE_SETPOINT,
                                 RESET_WRIST_LIMIT, Arena, _configuration_for, _joint_points)


def test_distinct_seeded_postures_have_identical_markers_and_different_command_effects():
    case = ReachCase((.41, .03), (.55, -.25))
    first = Arena(case, posture_seed=3)
    second = Arena(case, posture_seed=4)
    clone = Arena(case, posture_seed=3)
    np.testing.assert_array_equal(first.frame(), second.frame())
    np.testing.assert_array_equal(first.observation(), decode_markers(first.frame()))
    assert first._configuration == clone._configuration
    assert abs(first._configuration[1] - second._configuration[1]) > .30
    for _ in range(4):
        before, after = first.step((0., 0., 1.)), second.step((0., 0., 1.))
        for arena in (first, second):
            points = _joint_points(arena._configuration)
            np.testing.assert_allclose([math.dist(a, b) for a, b in zip(points, points[1:])], LINK_LENGTHS)
    assert math.dist(before[0], after[0]) > .02
    np.testing.assert_array_equal(before[1], after[1])
    assert first.render_rgb(scale=2).shape == (2 * CAMERA_SIZE, 2 * CAMERA_SIZE, 3)


def test_position_commands_are_relative_to_each_encoder_sample_and_servos_are_bounded():
    case = ReachCase((.52, -.30), (.25, .34))
    arena = Arena(case, max_steps=120)
    for malformed in ((1., 1.), (1., 0., 1., 0.), (math.inf, 0., 0.), (0., math.nan, 0.)):
        with pytest.raises(ValueError, match='three finite'):
            arena.step(malformed)
    assert arena.steps == 0
    np.testing.assert_allclose(arena.commanded_setpoint, arena._configuration)

    lower, upper = np.array(JOINT_LIMITS).T
    max_travel = TICK_SECONDS * np.asarray(MAX_JOINT_SPEED)
    bounded_action = np.array([1., -1., 1.])
    requested_delta = np.asarray(MAX_RELATIVE_SETPOINT) * bounded_action
    for tick in range(100):
        prior_angles = np.array(arena._configuration)
        arena.step(20 * bounded_action)
        np.testing.assert_allclose(arena.commanded_setpoint, np.clip(prior_angles + requested_delta, lower, upper))
        angles = np.array(arena._configuration)
        assert np.all(lower <= angles) and np.all(angles <= upper)
        assert np.all(np.abs(angles - prior_angles) <= max_travel + 1e-12)
        if tick == 0:
            assert np.all(np.abs(angles - np.asarray(arena.commanded_setpoint)) > .01)
    np.testing.assert_allclose(arena._configuration, (upper[0], lower[1], upper[2]))
    hold_angles = arena._configuration
    arena.step((0., 0., 0.))
    assert arena._configuration == hold_angles == arena.commanded_setpoint


@pytest.mark.parametrize('wrist', [-RESET_WRIST_LIMIT, 0., RESET_WRIST_LIMIT])
def test_full_shared_workspace_has_bounded_joint_solutions_at_reset_wrist_extremes(wrist):
    for x in np.linspace(*TASK_X, 11):
        for y in np.linspace(*TASK_Y, 11):
            configuration = _configuration_for((x, y), wrist)
            assert all(lo < angle < hi for angle, (lo, hi) in zip(configuration, JOINT_LIMITS))
            np.testing.assert_allclose(_joint_points(configuration)[-1], (x, y), atol=1e-12)


@pytest.mark.parametrize('posture_seed', [3, 4, 6])
def test_target_is_reachable_under_actuator_limits_at_each_test_posture(posture_seed):
    rng = np.random.default_rng(98823)
    corners = [(x, y) for x in TASK_X for y in TASK_Y]
    cases = [ReachCase(start, target) for start, target in zip(corners, corners[::-1])]
    cases += [sample_case(rng) for _ in range(12)]
    for case in cases:
        arena = Arena(case, posture_seed=posture_seed)
        np.testing.assert_allclose(arena.ground_truth_position, case.start, atol=1e-12)
        np.testing.assert_allclose(arena.observation(), (case.start, case.target), atol=.5 / CAMERA_CENTER)
        target_angles = _configuration_for(case.target, arena._configuration[2])
        while not arena.done:
            # Use a test-only joint oracle to verify the command interface can reach each target.
            error = np.subtract(target_angles, arena._configuration)
            arena.step(np.clip(error / MAX_RELATIVE_SETPOINT, -1, 1))
        assert arena.success
        assert all(math.dist(point, case.target) <= GOAL_TOLERANCE for point in arena.trace[-SUCCESS_TICKS:])
        errors = [math.dist(point, case.target) for point in arena.trace]
        errors += [errors[-1]] * (arena.max_steps - arena.steps)
        expected = np.average(errors[1:], weights=np.linspace(1, 2, arena.max_steps)) / errors[0]
        assert arena.weighted_normalized_error == pytest.approx(expected)


@pytest.mark.parametrize('start,target,expected', [(.400, .454, True), (.411, .467, False)])
def test_true_geometry_and_four_consecutive_ticks_control_termination(start, target, expected):
    arena = Arena(ReachCase((start, 0.), (target, 0.)), max_steps=SUCCESS_TICKS)
    public_position, public_target = arena.observation()
    assert (math.dist(public_position, public_target) > GOAL_TOLERANCE) is expected
    for _ in range(SUCCESS_TICKS):
        assert not arena.done
        arena.step((0., 0., 0.))
    assert arena.done and arena.success is expected
    assert arena.normalized_error == 1.
    with pytest.raises(RuntimeError, match='episode has ended'):
        arena.step((0., 0., 0.))
