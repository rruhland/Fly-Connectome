"""The whole predictive plan respects public rate and request limits."""

import numpy as np

import pytest

pytest.importorskip('mujoco')

from scripts.experimental.tt6_body_control import MOTOR_BUDGET
from scripts.experimental.tt6_bounded import _clipped_plan, _feasible, bounded_plan


def test_bounded_plan_reallocates_coupled_actuators():
    normal = np.array([[1., .8], [.8, 1.]])
    proposed = np.array([[.8, 0.]])
    requested = np.array([MOTOR_BUDGET - .15, 0.])
    chosen, fallback = bounded_plan(normal, proposed, requested, np.eye(2))
    assert not fallback
    np.testing.assert_allclose(chosen, [[.15, .52]], atol=1e-6)
    assert not np.allclose(chosen, _clipped_plan(proposed, requested))


def test_bounded_plan_respects_every_cumulative_prefix():
    requested = np.array([2.75, -3.05])
    proposed = np.array([[.8, -.4], [.7, -.7], [.6, -.8]])
    prefix = np.kron(np.tril(np.ones((3, 3))), np.eye(2))
    plan, fallback = bounded_plan(np.eye(6), proposed, requested, prefix)
    assert not fallback
    assert _feasible(plan, requested)
    np.testing.assert_allclose(np.sum(plan, axis=0) + requested, [MOTOR_BUDGET, -MOTOR_BUDGET])
    assert np.linalg.norm(plan - proposed) < np.linalg.norm(_clipped_plan(proposed, requested) - proposed)
