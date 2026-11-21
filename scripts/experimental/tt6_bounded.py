"""Bound every move of the camera-model predictive motor plan."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from .tt6_body_control import MOTOR_BUDGET
from .tt6_continuous import ContinualPlayer, PredictiveServo


_RIDGE = .00055


def _clipped_plan(plan, requested):
    clipped = np.empty_like(plan)
    state = requested.copy()
    for i, move in enumerate(plan):
        clipped[i] = np.clip(move, np.maximum(-1., -MOTOR_BUDGET - state),
                            np.minimum(1., MOTOR_BUDGET - state))
        state += clipped[i]
    return clipped


def _feasible(plan, requested, tolerance=1e-8):
    return (np.max(np.abs(plan)) <= 1. + tolerance and
            np.max(np.abs(requested + np.cumsum(plan, axis=0))) <= MOTOR_BUDGET + tolerance)


def _working_set(normal, center, initial, prefix, current):
    """Solve the strictly convex rate problem from a feasible starting plan."""
    size = len(center)
    identity = np.eye(size)
    constraint = np.vstack((identity, -identity, prefix, -prefix))
    limit = np.r_[np.ones(2 * size), MOTOR_BUDGET - current, MOTOR_BUDGET + current]
    projected = np.linalg.solve(normal, constraint.T)
    active = []
    candidate = initial.copy()
    for _ in range(4 * size + 8):
        direction = center - candidate
        if active:
            response = projected[:, active]
            rows = constraint[active]
            try:
                multiplier = np.linalg.solve(rows @ response, rows @ direction)
            except np.linalg.LinAlgError:
                return None
            direction -= response @ multiplier
        if np.max(np.abs(direction)) < 1e-9:
            if not active or np.min(multiplier) >= -1e-9:
                return candidate
            active.pop(int(np.argmin(multiplier)))
            continue
        slope = constraint @ direction
        blockers = slope > 1e-10
        blockers[active] = False
        ratios = np.divide(limit - constraint @ candidate, slope, out=np.full(len(limit), np.inf), where=blockers)
        blocking = int(np.argmin(ratios))
        step = float(np.clip(ratios[blocking], 0., 1.))
        candidate += step * direction
        if step < 1. - 1e-9:
            active.append(blocking)
    return None


def bounded_plan(normal, unconstrained, requested, prefix, *, solver='active'):
    """Minimize the same ridge objective with per-move and cumulative limits."""
    shape = unconstrained.shape
    if _feasible(unconstrained, requested):
        return unconstrained, False
    center = unconstrained.ravel()
    initial = _clipped_plan(unconstrained, requested)
    limits = np.tile(requested, shape[0])
    if solver == 'active':
        candidate = _working_set(normal, center, initial.ravel(), prefix, limits)
        if candidate is not None and _feasible(candidate.reshape(shape), requested):
            return _clipped_plan(candidate.reshape(shape), requested), False
    constraints = LinearConstraint(prefix, -MOTOR_BUDGET - limits, MOTOR_BUDGET - limits)

    def objective(candidate):
        change = candidate - center
        gradient = normal @ change
        return .5 * float(change @ gradient), gradient

    result = minimize(objective, initial.ravel(), method='SLSQP', jac=True,
                      bounds=Bounds(-1., 1.), constraints=(constraints,),
                      options=dict(ftol=1e-10, maxiter=35))
    candidate = result.x.reshape(shape)
    acceptable = _feasible(candidate, requested) and objective(result.x)[0] <= objective(initial.ravel())[0] + 1e-10
    if not acceptable:
        return initial, True
    # Snap away roundoff at actuator limits before applying the first rate.
    return _clipped_plan(candidate, requested), not result.success


class BoundedPredictiveServo(PredictiveServo):
    def __init__(self, fit, memory, *, learn=True, bounded=True, solver='active'):
        super().__init__(fit, memory, learn=learn)
        self.bounded = bounded
        self.solver = solver
        self.prefix = np.kron(np.tril(np.ones((self.moves, self.moves))), np.eye(fit.action_size))
        self.decisions = []

    def _proposal(self, goal, pitch):
        """Build the original predictive objective in public command coordinates."""
        fit = self.fit
        depth = fit.kernels.shape[1]
        future = []
        for step in range(self.horizon + 1):
            transient = fit.kernels[:, step:].T - 1.
            future.append(self.requested + np.sum(self.commands[:depth-step] * transient, axis=0))
        free = np.stack([*reversed(self.nominal_history[1:]), *future])
        delay = int(np.clip(round(self.observer.delay), 0, len(self.nominal_history) - 1))
        times = np.arange(1, self.horizon + 1)
        states = free[len(self.nominal_history) - 1 + times - delay]
        age = times[:, None] - np.arange(self.moves) - delay - 1
        effect = np.where(age[..., None] >= 0, fit.kernels.T[np.maximum(age, 0)], 0.)
        jacobian = fit.jacobian(self.effective) * self.weights[:, None]
        design = np.einsum('t,an,tmn->tamn', self.scale, jacobian, effect)
        design = design.reshape(4 * self.horizon, self.moves * fit.action_size)
        targets = np.tile(np.r_[goal, pitch], (self.horizon, 1))
        if self.intercept is not None:
            initial = np.clip(self.stroke * self.intercept, -.107, .145)
            targets[:, 1] += initial - np.clip(self.stroke * (self.intercept - times), -.107, .145)
        error = (targets - fit.pose(states) - self.correction) * self.weights * self.scale[:, None]
        normal = design.T @ design + _RIDGE * np.eye(design.shape[1])
        plan = np.linalg.solve(normal, design.T @ error.ravel()).reshape(self.moves, fit.action_size)
        return design, error.ravel(), normal, plan

    def command(self, goal, lead, pitch):
        design, target, normal, plan = self._proposal(goal, pitch)
        clipped = _clipped_plan(plan, self.requested)
        feasible = _feasible(plan, self.requested)
        selected, fallback = bounded_plan(normal, plan, self.requested, self.prefix, solver=self.solver)
        first = selected[0] if self.bounded else clipped[0]
        clipped_error = target - design @ clipped.ravel()
        selected_error = target - design @ selected.ravel()
        difference = clipped_error @ clipped_error + _RIDGE * np.sum(clipped**2)
        difference -= selected_error @ selected_error + _RIDGE * np.sum(selected**2)
        self.decisions.append(dict(intercept=self.intercept, infeasible=not feasible,
                                   first_clipped=bool(np.max(np.abs(plan[0] - clipped[0])) > 1e-7),
                                   at_budget=bool(np.max(np.abs(self.requested)) > MOTOR_BUDGET - .01),
                                   first_reallocated=float(np.linalg.norm(clipped[0] - selected[0])),
                                   objective_gain=float(difference), fallback=fallback))
        return first


class BoundedPredictivePlayer(ContinualPlayer):
    def __init__(self, adapter, prior, factory, response, timing, *, bounded=True, solver='active'):
        self.bounded = bounded
        self.solver = solver
        super().__init__(adapter, prior, factory, response, timing, predictive=True)

    def _make_servo(self):
        return BoundedPredictiveServo(self.adapter, self.timing_memory, bounded=self.bounded, solver=self.solver)

    def finish(self, reward):
        outcome = super().finish(reward)
        decisions = self.servo.decisions
        intercepts = [row for row in decisions if row['intercept'] is not None and -2 < row['intercept'] < 9]
        outcome['motor_plans'] = dict(decisions=len(decisions), intercept_decisions=len(intercepts),
                                      horizon_infeasible=sum(row['infeasible'] for row in intercepts),
                                      first_clipped=sum(row['first_clipped'] for row in intercepts),
                                      at_budget=sum(row['at_budget'] for row in intercepts),
                                      first_reallocated=sum(row['first_reallocated'] > .01 for row in intercepts),
                                      objective_gain=sum(row['objective_gain'] for row in intercepts),
                                      solver_warnings=sum(row['fallback'] for row in decisions))
        return outcome
