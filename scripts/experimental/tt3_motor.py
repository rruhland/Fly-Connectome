"""Learn a smooth visual body map from motion commands and camera tracks."""

from collections import deque
from itertools import combinations_with_replacement

import numpy as np

try:
    from .tt3_learning import MotorModel, TableLearner
except ImportError:
    from tt3_learning import MotorModel, TableLearner


COMMAND_SCALE = 8.
TERMS = tuple(term for degree in range(4) for term in combinations_with_replacement(range(3), degree))
POWERS = np.array([[term.count(axis) for axis in range(3)] for term in TERMS])
DEGREES = POWERS.sum(axis=1)
RIDGE = np.select((DEGREES == 0, DEGREES == 1, DEGREES == 2), (.01, .06, .20), default=.65)
DERIVATIVE_POWERS = np.maximum(POWERS[:, None, :] - np.eye(3, dtype=int), 0)


def _basis(coordinates):
    return np.power(np.asarray(coordinates)[..., None, :], POWERS).prod(axis=-1)


class VisualBodyModel:
    """Fit observed 3D position to integrated action coordinates about each reset."""

    def __init__(self, *, capacity=512):
        self.capacity = capacity
        self.cells = {}
        self.fitted = None
        self.changed = 0

    def snapshot(self):
        return dict(self.cells)

    def restore(self, cells):
        self.cells = dict(cells)
        self.fitted = None
        self.changed = 0

    def add(self, coordinates, position):
        if np.max(np.abs(coordinates)) > 2.7:
            return
        index = tuple(np.rint(coordinates * 5).astype(int))
        if index in self.cells:
            cached, observed, count = self.cells.pop(index)
            weight = 1. / min(8, count + 1)
            coordinates = cached + weight * (coordinates - cached)
            position = observed + weight * (position - observed)
            count += 1
        else:
            count = 1
        self.cells[index] = coordinates, position, count
        if len(self.cells) > self.capacity:
            del self.cells[next(iter(self.cells))]
        self.changed += 1

    def fit(self):
        if len(self.cells) < 10:
            return None
        if self.fitted is not None and self.changed < 5:
            return self.fitted
        coordinates, positions, support = zip(*self.cells.values())
        coordinates = np.stack(coordinates)
        positions = np.stack(positions)
        matrix = _basis(coordinates)
        weights = np.minimum(3., np.sqrt(support))
        regularization = np.diag(RIDGE)
        for _ in range(2):
            weighted = matrix * weights[:, None]
            coefficients = np.linalg.solve(matrix.T @ weighted + regularization, weighted.T @ positions)
            residual = np.linalg.norm(positions - matrix @ coefficients, axis=1)
            weights = np.minimum(3., np.sqrt(support)) / (1. + (residual / .012)**2)
        self.fitted = coefficients
        self.changed = 0
        return coefficients

    def pose(self, coordinates):
        coefficients = self.fit()
        return None if coefficients is None else _basis(coordinates) @ coefficients

    def derivative(self, coordinates):
        coefficients = self.fit()
        if coefficients is None:
            return None
        slopes = np.power(coordinates, DERIVATIVE_POWERS).prod(axis=-1) * POWERS
        return coefficients.T @ slopes / COMMAND_SCALE


class IntegratedMotor(MotorModel):
    """Combine visual differentiation with the body map where it predicts the current view."""

    def __init__(self, *, mix=.85, gain=.22, quiet=0.):
        super().__init__()
        self.body = VisualBodyModel()
        self.coordinates = np.zeros(3)
        self.mix = mix
        self.gain = gain
        self.quiet = quiet
        self.trust = 0.

    def matrix(self, position):
        local = super().matrix(position)
        pose = self.body.pose(self.coordinates)
        if pose is None:
            self.trust = 0.
            return local
        error = np.linalg.norm(position - pose)
        self.trust = self.mix / (1. + (error / .018)**2)
        fitted = self.body.derivative(self.coordinates)
        return local + self.trust * (fitted - local)

    def command(self, position, goal, *, jacobian=None):
        if self.gain == .22 and not self.quiet:
            return super().command(position, goal, jacobian=jacobian)
        correction = goal - position
        adjusted = position + correction * (self.gain / .22)
        if self.quiet and np.linalg.norm(correction) < self.quiet:
            return np.zeros(self.actions)
        return super().command(position, adjusted, jacobian=jacobian)

    def restore_local(self, state):
        restored = MotorModel.restore(state)
        self.positions = restored.positions
        self.commands = restored.commands
        self.responses = restored.responses
        self.samples = restored.samples


class IntegratedTableLearner(TableLearner):
    """TableLearner with a visual body map based only on its own command history."""

    def __init__(self, *args, mix=.85, gain=.22, quiet=0., **kwargs):
        super().__init__(*args, **kwargs)
        self.motor = IntegratedMotor(mix=mix, gain=gain, quiet=quiet)

    def reset(self, option=None):
        super().reset(option)
        self.motor.coordinates = np.zeros(3)
        self.coordinate_history = deque(maxlen=self.history.maxlen)
        self.starting_body = self.motor.body.snapshot()
        self.accept_body = True

    def act(self, frames):
        if self.action is not None:
            self.motor.coordinates = self.motor.coordinates + self.action / COMMAND_SCALE
        self.coordinate_history.append(self.motor.coordinates)
        return super().act(frames)

    def _update_motor(self, controlled):
        new_binding = controlled.id != self.identity
        if new_binding and self.identity is not None:
            self.motor.restore_local(self.starting_motor)
            self.motor.body.restore(self.starting_body)
            self.identity = None
        super()._update_motor(controlled)
        if new_binding:
            observations = list(zip(self.coordinate_history, self.history))
            anchor = self.motor.body.pose(np.zeros(3))
            if anchor is not None and self.ready is not None:
                self.accept_body = np.linalg.norm(anchor - self.ready) < .053
        else:
            observations = [(self.coordinate_history[-1], self.history[-1])]
        if self.accept_body:
            for coordinates, row in observations:
                position = row.positions.get(controlled.id)
                if position is not None:
                    self.motor.body.add(coordinates, position)

    def finish(self):
        result = super().finish()
        result['body_cells'] = len(self.motor.body.cells)
        return result
