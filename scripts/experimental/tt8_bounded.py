"""Enforce rate and motor travel constraints across the predictive servo's plan."""

from functools import lru_cache
from time import process_time_ns

import numpy as np

from .tt6_body_control import MOTOR_BUDGET
from .tt6_bounded import _clipped_plan, bounded_plan
from .tt6_continuous import ContinualPlayer, PredictiveServo


@lru_cache(maxsize=8)
def _prefix(moves, axes):
    return np.kron(np.tri(moves), np.eye(axes))


def solve_bounded_plan(normal, rhs, requested):
    """Return the feasible minimizer and solver measurements for one control frame."""
    axes = len(requested)
    moves = len(rhs) // axes
    unconstrained = np.linalg.solve(normal, rhs).reshape(moves, axes)
    future = requested + np.cumsum(unconstrained, axis=0)
    violation = float(max(0., np.max(np.abs(unconstrained)) - 1., np.max(np.abs(future)) - MOTOR_BUDGET))
    feasible, warning = bounded_plan(normal, unconstrained, requested, _prefix(moves, axes))
    sequential = _clipped_plan(unconstrained, requested)
    change = (sequential - feasible).ravel()
    midpoint = ((sequential + feasible) * .5).ravel()
    improvement = float(change @ (normal @ midpoint - rhs))
    planned = requested + np.cumsum(feasible, axis=0)
    active = bool(np.max(np.abs(feasible)) >= 1. - 1e-8 or np.max(np.abs(planned)) >= MOTOR_BUDGET - 1e-8)
    changed = bool(np.max(np.abs(feasible[0] - sequential[0])) > 1e-7)
    return feasible, dict(violation=violation, active=active, improvement=max(0., improvement),
                          changed=changed, warning=warning)


class BoundedPlanMixin:
    """Use the same bound solver after any PredictiveServo strike objective."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan_evidence = dict(frames=0, infeasible_frames=0, active_frames=0, altered_first_moves=0,
                                  solver_warning_frames=0, cpu_ns=0, total_violation=0.,
                                  objective_improvement=0.)

    def _solve_plan(self, normal, rhs):
        began = process_time_ns()
        plan, evidence = solve_bounded_plan(normal, rhs, self.requested)
        result = self.plan_evidence
        result['cpu_ns'] += process_time_ns() - began
        result['frames'] += 1
        result['infeasible_frames'] += evidence['violation'] >= 1e-9
        result['active_frames'] += evidence['active']
        result['altered_first_moves'] += evidence['changed']
        result['solver_warning_frames'] += evidence['warning']
        result['total_violation'] += evidence['violation']
        result['objective_improvement'] += evidence['improvement']
        return plan[0]


class BoundedServo(BoundedPlanMixin, PredictiveServo):
    """Predictive servo with limits on every proposed move."""


class BoundedPlayer(ContinualPlayer):
    """Use a bounded predictive plan for the existing receiving player."""

    def _make_servo(self):
        return BoundedServo(self.adapter, self.timing_memory, learn=self.adjust)

    def finish(self, reward):
        result = super().finish(reward)
        result['motor_plan'] = self.servo.plan_evidence
        return result
