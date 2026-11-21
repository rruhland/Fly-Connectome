"""Infer camera-space manipulator control from the commands of live play."""

from collections import deque

import numpy as np
from scipy.optimize import lsq_linear

from .tt6_body_control import _response_bank


TRAVEL = 5.25
ORIENTATION = .105
LAGS = (0, 1, 2, 3)


def kernel_for(body=None):
    if body is not None:
        return np.median(body.kernels, axis=0)
    bank, delays = _response_bank()
    return bank[np.argmin(abs(delays - .48))]


def blade_pose(tracker):
    actor = tracker.actor
    if actor is None or not actor.observed:
        return None
    normal = getattr(tracker, 'actor_normal', None)
    if normal is not None:
        return np.r_[actor.position, ORIENTATION * np.asarray(normal)[[2, 0]]]
    pitch = tracker.actor_pitch
    return np.r_[actor.position, np.nan if pitch is None else ORIENTATION * np.sin(pitch), np.nan]


class InverseModel:
    """Fit absolute poses or optical secants with delayed input alternatives."""

    def __init__(self, width, kernel, mode, seed, *, solver='bounded', state=None):
        self.width = width
        self.mode = mode
        self.solver = solver if state is None else state.get('solver', 'clipped')
        self.kernel = np.asarray(kernel)
        self.rng = np.random.default_rng(seed)
        self.axis = np.triu_indices(width)
        size = width + len(self.axis[0])
        self.ridge = np.r_[np.full(width, .14), np.full(size - width, 12.)]
        self.normal = np.tile(np.diag(self.ridge), (len(LAGS), 3, 1, 1))
        self.cross = np.zeros((len(LAGS), size, 5))
        self.weights = np.zeros_like(self.cross)
        self.cost = np.zeros(len(LAGS))
        self.samples = self.updates = 0
        self.lag = 0
        self.learning = True
        if state is not None:
            for key in ('normal', 'cross', 'weights', 'cost'):
                setattr(self, key, np.asarray(state[key]))
            for key in ('samples', 'updates', 'lag', 'learning'):
                setattr(self, key, state[key])
            self.rng.bit_generator.state = state['rng']
        self.reset()

    def reset(self):
        self.history = deque(maxlen=16)
        self.recent = np.zeros((len(self.kernel), self.width))
        self.request = np.zeros(self.width)
        self.q = deque([self.request] * len(LAGS), maxlen=16)
        self.home = self.anchor = self.last = self.previous = None
        self.orientation = [None, None]
        self.tick = 0
        self.dither = np.zeros(self.width)
        self.exploration = self.error = 0.

    def features(self, q):
        left, right = self.axis
        return np.concatenate([q, q[..., left] * q[..., right]], axis=-1)

    def jacobian(self, q):
        left, right = self.axis
        model = self.weights[self.lag]
        j = model[:self.width].copy()
        np.add.at(j, left, model[self.width:] * q[right, None])
        np.add.at(j, right, model[self.width:] * q[left, None])
        return j.T

    def free(self, horizon):
        return self.request + (self.kernel[horizon:] - 1.) @ self.recent[:len(self.kernel) - horizon]

    def advance(self, previous):
        if previous is None:
            return
        applied = np.clip(self.request + previous, -TRAVEL, TRAVEL)
        self.recent[1:] = self.recent[:-1]
        self.recent[0] = applied - self.request
        self.request = applied
        self.q.appendleft(self.free(0))
        self.tick += 1

    def measure(self, pose):
        if pose is None:
            return
        if self.home is None:
            self.home = pose.copy()
            self.anchor = np.asarray(self.q)[:len(LAGS)].copy()
        if self.last is not None:
            tick, _, old = self.last
            if np.linalg.norm(pose[:3] - old[:3]) > .09 * (self.tick - tick) + .037:
                return
        states = np.asarray(self.q)[:len(LAGS)].copy()
        if self.learning:
            if self.mode == 'pose':
                self._sample(states, pose, self.anchor, self.home)
            else:
                for tick, earlier, seen in self.history:
                    if self.tick - tick in (3, 6):
                        self._sample(states, pose, earlier, seen)
        for axis in range(2):
            if np.isfinite(pose[3 + axis]):
                self.orientation[axis] = states, pose[3 + axis]
        self.previous = self.last
        self.last = (self.tick, states, pose)
        self.history.append(self.last)

    def _sample(self, states, pose, earlier, seen):
        features = self.features(states) - self.features(earlier)
        if np.max(abs(features[:, :self.width])) < .039:
            return
        change = pose - seen
        valid = np.isfinite(change)
        residual = change - np.einsum('li,liw->lw', features, self.weights)
        xyz = np.linalg.norm(residual[:, :3], axis=1)
        self.error = float(xyz[self.lag])
        self.cost = .966 * self.cost + .034 * np.minimum(.22, xyz)
        weight = 1 / np.sqrt(1. + (xyz / .085)**2)
        self.samples += 1
        for group, selected in enumerate((slice(0, 3), slice(3, 4), slice(4, 5))):
            if not valid[selected].all():
                continue
            self.normal[:, group] += weight[:, None, None] * features[:, :, None] * features[:, None, :]
            self.cross[:, :, selected] += weight[:, None, None] * features[:, :, None] * change[selected]
        if self.samples % 5 == 0:
            self.weights[:, :, :3] = np.linalg.solve(self.normal[:, 0], self.cross[:, :, :3])
            for group, index in ((1, 3), (2, 4)):
                fitted = np.linalg.solve(self.normal[:, group], self.cross[:, :, index:index + 1])
                self.weights[:, :, index] = fitted[:, :, 0]
            if self.samples >= 32:
                cost = self.cost + .0017 * np.arange(len(LAGS))
                best = int(np.argmin(cost))
                if cost[best] + .0014 < cost[self.lag]:
                    self.lag = best
            self.updates += 1

    def live_pose(self):
        if self.last is None:
            return None
        _, observed_state, observed_pose = self.last
        model = self.weights[self.lag]
        displacement = (self.features(self.q[self.lag]) - self.features(observed_state[self.lag])) @ model
        pose = observed_pose + displacement
        for axis, seen in enumerate(self.orientation):
            if seen is not None and not np.isfinite(pose[3 + axis]):
                states, value = seen
                change = self.features(self.q[self.lag]) - self.features(states[self.lag])
                pose[3 + axis] = value + np.clip(change @ model[:, 3 + axis], -.040, .040)
        return np.where(np.isfinite(pose), pose, 0.)

    def control(self, target, deadline, *, explore, fixed=(), strike_rate=0.):
        if self.last is None:
            return np.zeros(self.width)
        lag = LAGS[self.lag]
        q = self.q[lag]
        jac = self.jacobian(q)
        position_rank = np.linalg.svd(jac[:3], compute_uv=False)[-1]
        horizon = int(np.clip(np.ceil(deadline), lag + 3, lag + 6)) if deadline is not None else lag + 5
        current = self.live_pose()
        wanted = target(horizon)
        projected = (self.features(self.free(max(1, horizon - lag))) - self.features(q)) @ self.weights[self.lag]
        clipped = self.solver in ('clipped', 'register')
        forecast = current + (1. if clipped else .72) * projected
        if not clipped and self.previous is not None and self.tick - self.last[0] <= 1:
            interval = self.last[0] - self.previous[0]
            if 0 < interval <= 3:
                movement = (self.last[2][:3] - self.previous[2][:3]) / interval
                forecast[:3] += .34 * min(3, horizon) * np.clip(movement, -.050, .050)
        influence = self.kernel[:horizon - lag].sum()
        matrix = influence * jac
        orientation = np.array([1., 1., 1., 1.35, .75])
        matrix *= orientation[:, None]
        error = np.clip(wanted - forecast, -.20, .20) * orientation
        if self.solver == 'drive' and deadline is not None and lag + .6 < deadline < lag + 7:
            arrival = horizon - lag
            weight = 2.8 * np.exp(-.5 * ((deadline - horizon) / 2.)**2)
            motion = jac[1] @ (self.free(arrival) - self.free(arrival - 1))
            matrix = np.vstack((matrix, weight * self.kernel[arrival - 1] * jac[1]))
            error = np.r_[error, weight * (strike_rate - motion)]
        penalty = .00066 + .00200 * (np.abs(self.request) / TRAVEL)**4
        cap = .85 if deadline is not None and -2. < deadline < 7. else .59
        lower = np.maximum(-cap, -TRAVEL - self.request)
        upper = np.minimum(cap, TRAVEL - self.request)
        if fixed:
            lower[list(fixed)] = upper[list(fixed)] = 0.
        if clipped:
            command = np.linalg.solve(matrix.T @ matrix + np.diag(penalty),
                                      matrix.T @ error - .00020 * self.request)
            bounded = np.clip(command, lower, upper)
        else:
            enabled = upper - lower > 1e-8
            design = np.vstack([matrix, np.diag(np.sqrt(penalty))])[:, enabled]
            rhs = np.r_[error, -.00035 * self.request / np.sqrt(penalty)]
            solution = np.linalg.solve(design.T @ design, design.T @ rhs)
            if np.any((solution < lower[enabled]) | (solution > upper[enabled])):
                solution = lsq_linear(design, rhs, bounds=(lower[enabled], upper[enabled]), method='bvls').x
            bounded = np.zeros(self.width)
            bounded[enabled] = solution
            command = bounded
        if not explore:
            return bounded
        if self.tick % 3 == 0:
            gram = self.normal[self.lag, 0, :self.width, :self.width]
            draw = np.linalg.solve(np.linalg.cholesky(gram).T, self.rng.normal(size=self.width))
            draw[list(fixed)] = 0.
            self.dither = draw / max(.01, np.max(abs(draw)))
        initial = np.clip((.012 - position_rank) / .012, 0., 1.)
        budget = .020 + .44 * initial + .13 / (1. + self.samples / 65)
        if deadline is not None and -3. < deadline < 8.:
            budget *= .16 + .40 * initial
        output = np.clip(command + budget * self.dither, lower, upper)
        self.exploration += float(np.sum(abs(output - bounded)))
        return output

    def state(self):
        return dict(normal=self.normal.tolist(), cross=self.cross.tolist(), weights=self.weights.tolist(),
                    cost=self.cost.tolist(), rng=self.rng.bit_generator.state, samples=self.samples,
                    updates=self.updates, lag=self.lag, learning=self.learning, kernel=self.kernel.tolist(),
                    mode=self.mode, solver=self.solver)
