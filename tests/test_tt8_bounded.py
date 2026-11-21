"""Check that full-plan limits honor the caller's strike objective."""

import numpy as np
import pytest

pytest.importorskip('mujoco')

from scripts.experimental.tt6_body_control import MOTOR_BUDGET, CameraBody
from scripts.experimental.tt6_continuous import PredictiveServo
from scripts.experimental.tt6_timing import TimingMemory
from scripts.experimental.tt8_bounded import BoundedPlanMixin, solve_bounded_plan


def test_strike_objective_is_constrained_by_each_move():
    class StrikeServo(PredictiveServo):
        def _strike_objective(self, normal, rhs, states, effect, jac, targets):
            rhs[::2] += .00055 * .8

    class BoundedStrikeServo(BoundedPlanMixin, StrikeServo):
        pass

    fit = CameraBody(np.zeros((6, 4)), np.zeros(2), np.ones((2, 14)), .01, 200, 100, True)
    legacy, bounded = [cls(fit, TimingMemory()) for cls in (StrikeServo, BoundedStrikeServo)]
    for servo in (legacy, bounded):
        servo.requested[0] = MOTOR_BUDGET - .15
    goal = np.zeros(3)
    np.testing.assert_allclose(legacy.command(goal, goal, 0.), [.15, 0.])
    np.testing.assert_allclose(bounded.command(goal, goal, 0.), [.05, 0.])
    assert bounded.plan_evidence['infeasible_frames'] == 1
    assert bounded.plan_evidence['altered_first_moves'] == 1
    assert bounded.plan_evidence['objective_improvement'] > 0.
    assert bounded.plan_evidence['solver_warning_frames'] == 0


def test_feasible_quadratic_keeps_its_full_solution():
    rhs = np.array([.1, .2, -.3, -.4, .5, .6])
    plan, evidence = solve_bounded_plan(np.eye(6), rhs, np.zeros(2))
    np.testing.assert_array_equal(plan.ravel(), rhs)
    assert evidence == dict(violation=0., active=False, improvement=0., changed=False, warning=False)
