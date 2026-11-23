"""Learn local action-conditioned visual consequences within the planar task."""

from collections import deque
from itertools import product

import numpy as np

from .tt24_planar_learning import RATE_VALUES


class LocalFit:
    """Estimate optical displacements from causal observations and bounded probes."""

    def __init__(self, inputs):
        self.weights = np.zeros((inputs, 2))
        self.variance = np.eye(inputs) * 5.
        self.samples = 0

    def update(self, feature, observed):
        error = np.clip(observed - feature @ self.weights, -.8, .8)
        influence = self.variance @ feature
        update = influence / (.997 + feature @ influence)
        self.weights += np.outer(update, error)
        self.variance = (self.variance - np.outer(update, influence)) / .997
        self.samples += 1


class OpticalEffects:
    """Use issued rate and measured image motion to propose statistical optical futures."""

    def __init__(self, axes, *, source_gain=4., egocentric=False):
        if egocentric:
            independent = np.eye(axes, dtype=int)
            self.action_index = np.r_[np.zeros((1, axes), dtype=int), independent, 2 * independent]
        else:
            self.action_index = np.asarray(list(product(range(len(RATE_VALUES)), repeat=axes)))
        self.actions = RATE_VALUES[self.action_index]
        self.motor = LocalFit(3 + axes)
        self.mover = LocalFit(5 + axes if egocentric else 5)
        self.egocentric = egocentric
        self.source_gain = source_gain
        self.axes = axes
        self.reset()

    def reset(self):
        self.positions = deque(maxlen=4)
        self.predictions = self.supported = 0

    def initial(self, reading):
        self.positions.clear()
        if reading.appearance is not None:
            self.positions.append(reading.appearance)

    def _history(self):
        actor, ball, span = zip(*self.positions, strict=True)
        low = np.asarray(actor)
        high = np.asarray(ball)
        scale = float(np.median(span))
        return low, high, scale

    def observed(self, action, reading, *, plastic):
        next_appearance = reading.appearance
        if next_appearance is None:
            self.positions.clear()
            return
        if plastic and len(self.positions) >= 3:
            actor, ball, scale = self._history()
            new_actor, new_ball, _ = next_appearance
            actor_flow = (actor[-1] - actor[-2]) / scale
            ball_flow = np.diff(ball[-3:], axis=0).ravel() / scale
            self.motor.update(np.r_[1., action, actor_flow], (new_actor - actor[-1]) / scale)
            mover_feature = np.r_[1., ball_flow, action] if self.egocentric else np.r_[1., ball_flow]
            self.mover.update(mover_feature, (new_ball - ball[-1]) / scale)
        self.positions.append(next_appearance)

    def hint(self, prior, dt, *, horizon=2):
        if len(self.positions) < 4 or min(self.motor.samples, self.mover.samples) < 5:
            return None
        actor, ball, scale = self._history()
        actor_flow = np.broadcast_to((actor[-1] - actor[-2]) / scale, (len(self.actions), 2))
        ball_flow = np.diff(ball[-3:], axis=0) / scale
        actor_last = np.broadcast_to(actor[-1], (len(self.actions), 2))
        ball_last = ball[-1]
        shape = (len(self.actions), len(actor), 2)
        actor_paths = [np.broadcast_to(actor, shape)]
        ball_paths = [np.broadcast_to(ball, shape)]
        if self.egocentric:
            ball_flow = np.broadcast_to(ball_flow, (len(self.actions), 2, 2))
        for _ in range(horizon):
            motor_feature = np.column_stack((np.ones(len(self.actions)), self.actions, actor_flow))
            actor_flow = np.clip(motor_feature @ self.motor.weights, -.65, .65)
            if self.egocentric:
                n = len(self.actions)
                mover_feature = np.column_stack((np.ones(n), ball_flow.reshape(n, 4), self.actions))
                planned = np.clip(mover_feature @ self.mover.weights, -.7, .7)
                ball_flow = np.stack((ball_flow[:, -1], planned), axis=1)
            else:
                planned = np.clip(np.r_[1., ball_flow.ravel()] @ self.mover.weights, -.7, .7)
                ball_flow = np.stack((ball_flow[-1], planned))
            actor_last = actor_last + scale * actor_flow
            ball_last = ball_last + scale * planned
            actor_paths.append(actor_last[:, None, :])
            ball_paths.append(np.broadcast_to(ball_last.reshape(-1, 1, 2), (len(self.actions), 1, 2)))
        actor_path = np.concatenate(actor_paths, axis=1)
        ball_path = np.concatenate(ball_paths, axis=1)
        result = prior.score_history(actor_path[:, -5:], ball_path[:, -5:], dt=dt, scale=scale, details=True)
        probability = result['score']
        self.predictions += len(probability)
        self.supported += int(np.count_nonzero(result['support']))
        if self.egocentric:
            return self.source_gain * (np.column_stack((np.full(self.axes, probability[0]),
                                                        probability[1:1 + self.axes], probability[1 + self.axes:]))
                                       - probability[0])
        return self.source_gain * np.array([[np.mean(probability[self.action_index[:, axis] == j]) - np.mean(probability)
                                            for j in range(len(RATE_VALUES))] for axis in range(self.axes)])
