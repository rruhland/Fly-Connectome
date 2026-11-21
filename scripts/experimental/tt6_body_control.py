"""Learn an RGB-to-action body model without joint types, geometry or state."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import lfilter

from .tt5_player import RallyPlayer


POSE_WEIGHT = np.array((1., 1., 1., .12))
MOTOR_BUDGET = 3.85  # Public probe excursion in intervals at full command.
RESPONSE_FRAMES = 14


def _features(state):
    left, right = np.triu_indices(state.shape[-1])
    return np.concatenate((np.ones((*state.shape[:-1], 1)), state, state[..., left] * state[..., right]), axis=-1)


def _regression(design, values, *, penalty):
    weights = np.ones(len(values))
    for _ in range(4):
        weighted = design * weights[:, None]
        fit = np.linalg.solve(weighted.T @ weighted + np.diag(penalty), weighted.T @ (values * weights))
        residual = design @ fit - values
        weights = (1. + (residual / .034)**2) ** -.50
    return fit


def _response_bank():
    """Sample plausible settling responses to one interval of constant motor rate."""
    time = np.arange(RESPONSE_FRAMES)[:, None] + (np.arange(9) + .5) / 9
    responses = []
    for frequency in (.23, .31, .41, .54, .70, .88, 1.10, 1.40, 1.80):
        for damping in (.44, .61, .78, .93, 1.08, 1.28):
            root = frequency * np.lib.scimath.sqrt(damping**2 - 1.)
            error = np.exp(-frequency * damping * time) * (
                np.cosh(root * time) + frequency * damping * np.sinh(root * time) / root)
            responses.append(1. - error.real.mean(axis=1))
    kernels = np.asarray(responses)
    # Expose a scalar effective delay for sharing across mechanisms. The fitted
    # curve retains onset, overshoot and settling within the local map.
    delay = np.maximum(np.sum(1. - kernels, axis=1) - .5, 0.)
    return kernels, delay / (1. + delay)


@dataclass(frozen=True)
class CameraBody:
    coefficients: np.ndarray
    lags: np.ndarray
    kernels: np.ndarray
    error: float
    samples: int
    pitch_samples: int
    pitch_controllable: bool

    @property
    def ready(self):
        return self.home[:3]

    @property
    def home(self):
        return self.coefficients[0]

    @property
    def action_size(self):
        return len(self.lags)

    def pose(self, state):
        return _features(state) @ self.coefficients

    def jacobian(self, state):
        size = self.action_size
        left, right = np.triu_indices(size)
        jacobian = self.coefficients[1:1 + size].copy()
        cross = self.coefficients[1 + size:]
        np.add.at(jacobian, left, state[right, None] * cross)
        np.add.at(jacobian, right, state[left, None] * cross)
        return jacobian.T

    def without_channel(self, index):
        size = self.action_size
        left, right = np.triu_indices(size)
        mask = np.r_[False, np.arange(size) == index, (left == index) | (right == index)]
        return replace(self, coefficients=np.where(mask[:, None], 0., self.coefficients))

    def reversed_channels(self):
        # Commands are reflected in the fitted model, preserving contact/flight experience.
        coefficients = self.coefficients.copy()
        coefficients[1:1 + self.action_size] *= -1.
        return replace(self, coefficients=coefficients)

    def stats(self):
        sensitivity = self.jacobian(np.zeros(self.action_size))
        return dict(ready=self.ready.tolist(), home_pitch=float(self.home[3]),
                    response_per_command=sensitivity.tolist(), lags=self.lags.tolist(),
                    settling_curves=self.kernels.tolist(),
                    median_fit_error=self.error, motor_samples=self.samples, pitch_samples=self.pitch_samples,
                    pitch_controllable=self.pitch_controllable, rank=int(np.linalg.matrix_rank(sensitivity[:3])))


def _samples(histories, tracker_factory):
    width = len(histories[0][0][0])
    length = max(map(len, histories))
    commands = np.zeros((len(histories), length, width))
    episodes, ticks, positions = [], [], []
    for index, history in enumerate(histories):
        tracker = tracker_factory()
        for tick, (action, frames) in enumerate(history):
            commands[index, tick] = action
            tracker.update(frames)
            if tracker.actor is not None and tracker.actor.observed:
                episodes.append(index)
                ticks.append(tick)
                positions.append((*tracker.actor.position, np.nan if tracker.actor_pitch is None else tracker.actor_pitch))
    return commands, (np.array(episodes, dtype=int), np.array(ticks, dtype=int)), np.array(positions)


def fit_camera_body(histories, tracker_factory, *, lag_prior=None):
    """Fit contact pose to cumulative RGB probe commands, with optional old lag evidence."""
    commands, indices, poses = _samples(histories, tracker_factory)
    width = commands.shape[-1]
    if len(poses) < 12 * width:
        raise ValueError('insufficient observed paddle poses in the RGB motor probes')
    kernels, grid = _response_bank()
    cumulative = np.cumsum(commands, axis=1)[indices]
    responses = np.array([cumulative + lfilter(kernel - 1., [1.], commands, axis=1)[indices] for kernel in kernels])
    lags = np.full(width, np.argmin(abs(grid - (.56 if lag_prior is None else lag_prior))))
    cols = np.arange(width)
    xyz = poses[:, :3]
    valid = np.isfinite(poses[:, 3])
    pitches = poses[valid, 3]
    pitch_varies = len(pitches) >= 35 and np.std(pitches) > .075
    height = width + 1 + width * (width + 1) // 2
    penalty = np.r_[.00001, np.full(width, .06), np.full(height - width - 1, 9.)]

    def fit(selection, *, regularized):
        states = responses[selection, :, cols].T
        design = _features(states)
        outputs = [_regression(design, target, penalty=penalty) for target in xyz.T]
        estimate = design @ np.array(outputs).T
        errors = np.linalg.norm(estimate - xyz, axis=1)
        score = np.mean(np.minimum(errors, .11)**2)
        if pitch_varies:
            coeffs = _regression(design[valid], pitches * POSE_WEIGHT[3], penalty=penalty)
            pitch_error = np.abs(design[valid] @ coeffs - pitches * POSE_WEIGHT[3])
            score += .75 * np.mean(np.minimum(pitch_error, .040)**2)
            outputs.append(coeffs / POSE_WEIGHT[3])
        else:
            outputs.append(np.r_[.27 if not len(pitches) else np.median(pitches), np.zeros(height - 1)])
        if regularized and lag_prior is not None:
            score += .0011 * np.mean((grid[selection] - lag_prior)**2)
        return score, np.array(outputs).T, errors

    for _ in range(2):
        for channel in range(width):
            candidates = np.tile(lags, (len(grid), 1))
            candidates[:, channel] = np.arange(len(grid))
            lags[channel] = min(range(len(grid)), key=lambda i: fit(candidates[i], regularized=True)[0])
    _, coefficients, errors = fit(lags, regularized=False)
    gain = coefficients[1:1 + width].T
    if np.linalg.svd(gain[:3], compute_uv=False)[-1] < .014 or np.median(errors) > .075:
        raise ValueError('the RGB motor probes did not identify three reachable position directions')
    return CameraBody(coefficients, grid[lags], kernels[lags], float(np.median(errors)), len(poses), len(pitches),
                      bool(np.max(np.abs(gain[3])) >= .035))


class CameraServo:
    """Track actuator lag in public command space and correct camera pose residuals."""

    def __init__(self, fit):
        self.fit = fit
        self.requested = np.zeros(fit.action_size)
        self.effective = self.requested.copy()
        self.commands = np.zeros((RESPONSE_FRAMES, fit.action_size))
        self.correction = np.zeros(4)
        self.pose = fit.ready.copy()
        self.pitch = fit.home[3]
        self.velocity = np.zeros(3)

    def advance(self, preceding):
        if preceding is not None:
            next_requested = np.clip(self.requested + preceding, -MOTOR_BUDGET, MOTOR_BUDGET)
            self.commands[1:] = self.commands[:-1]
            self.commands[0] = next_requested - self.requested
            self.requested = next_requested
            self.effective = self.requested + np.sum(self.commands * (self.fit.kernels.T - 1.), axis=0)

    def observe(self, actor, pitch):
        predicted = self.fit.pose(self.effective)
        if actor is not None and actor.observed:
            delta = np.asarray(actor.position) - predicted[:3]
            if np.linalg.norm(delta) < .16:
                self.correction[:3] = .45 * self.correction[:3] + .55 * np.clip(delta, -.07, .07)
        if pitch is not None and np.isfinite(pitch) and abs(pitch - predicted[3]) < .34:
            self.correction[3] = .65 * self.correction[3] + .35 * (pitch - predicted[3])
        observed = predicted + self.correction
        self.velocity = .35 * self.velocity + .65 * (observed[:3] - self.pose)
        self.pose, self.pitch = observed[:3], observed[3]

    def command(self, goal, lead, pitch):
        baseline = self.requested + np.sum(self.commands[:-2] * (self.fit.kernels[:, 2:].T - 1.), axis=0)
        target = np.r_[lead + .72 * (lead - goal), pitch]
        error = target - self.fit.pose(baseline) - self.correction
        jacobian = self.fit.jacobian(baseline) * self.fit.kernels[:, 1]
        matrix = np.vstack((POSE_WEIGHT[:, None] * jacobian, .009 * np.eye(self.fit.action_size)))
        residual = np.r_[POSE_WEIGHT * error, np.zeros(self.fit.action_size)]
        lower = np.maximum(-1., -MOTOR_BUDGET - self.requested) - 1e-9
        upper = np.minimum(1., MOTOR_BUDGET - self.requested) + 1e-9
        return np.clip(lsq_linear(matrix, residual, bounds=(lower, upper), method='bvls').x, -1., 1.)


class CameraBodyPlayer(RallyPlayer):
    """Use the same visual ball/stroke policy with a newly identified motor map."""

    def _make_servo(self):
        return CameraServo(self.adapter)

    def _reachable_plane(self, plane, ball):
        return plane

    def _wrist_available(self):
        return self.adapter.pitch_controllable
