"""Identify a stereo racket's signed motor response and servo it in world space."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.signal import lfilter

from .tt4_transfer import ArmAdapter, CalibrationSample


# Rounded measurements from the visible mount and two approximately equal links.
MOUNT = (0., -1.90, .24)
LINK = .40
# Conservative ranges for the elbow-folded branch, in radians.
ANGLE_LOW = np.array([-1.10, -1.10, .48, -.60])
ANGLE_HIGH = np.array([1.10, .61, 2.28, .73])
HOME_PITCH = .20


def angles(positions, geometry):
    """Convert blade centers into the three angles of a foldable two-link arm."""
    x, y, z = np.moveaxis(np.asarray(positions) - geometry[:3], -1, 0)
    radius = np.hypot(x, y)
    half_elbow = np.arccos(np.clip(np.hypot(radius, z) / (2. * geometry[3]), .03, .999))
    return np.stack((np.arctan2(x, y), np.arctan2(z, radius) - half_elbow, 2. * half_elbow), axis=-1)


def centers(joints, geometry):
    yaw, shoulder, elbow = np.moveaxis(joints[..., :3], -1, 0)
    radius = geometry[3] * (np.cos(shoulder) + np.cos(shoulder + elbow))
    height = geometry[3] * (np.sin(shoulder) + np.sin(shoulder + elbow))
    return np.stack((radius * np.sin(yaw), radius * np.cos(yaw), height), axis=-1) + geometry[:3]


def _filtered(commands, lag):
    return lfilter([1. - lag], [1., -lag], np.cumsum(commands, axis=1), axis=1)


def _body_observations(histories, factory):
    trajectories, selected = [], []
    for history in histories:
        tracker = factory()
        samples, direct = [], []
        for action, frames in history:
            tracks = tracker.update(frames)
            current = {t.id: np.asarray(t.position) for t in tracks if t.observed}
            samples.append(CalibrationSample(action, current, dict(tracker.appearances)))
            actor = getattr(tracker, 'actor', None)
            pitch = getattr(tracker, 'actor_pitch', None)
            direct.append((None, None) if actor is None or not actor.observed else (actor.position, pitch))
        trajectories.append(samples)
        selected.append(direct)
    if sum(position is not None for row in selected for position, _ in row) < 55:
        fallback = ArmAdapter(trajectories)
        selected = []
        for rows in trajectories:
            requested = effective = np.zeros(4)
            picks = []
            for sample in rows:
                requested = requested + sample.action / 4.
                effective = fallback.lag * effective + (1. - fallback.lag) * requested
                target = fallback.pose(effective)
                candidates = [point for key, point in sample.positions.items()
                              if sample.appearances.get(key) == fallback.hue or fallback.hue is None]
                point = min(candidates, key=lambda p: np.linalg.norm(p - target), default=None)
                picks.append((point, None) if point is not None and np.linalg.norm(point - target) < .16
                             else (None, None))
            selected.append(picks)
    return selected


def _linear_fit(predictor, target):
    design = np.column_stack((np.ones(len(predictor)), predictor))
    weights = np.ones(len(target))
    for _ in range(3):
        model = np.linalg.lstsq(design * weights[:, None], target * weights, rcond=None)[0]
        residual = design @ model - target
        weights = 1. / np.sqrt(1. + (residual / .12)**2)
    return model, np.mean(np.minimum(residual**2, .30**2))


@dataclass(frozen=True)
class FittedArm:
    geometry: np.ndarray
    scripts: np.ndarray
    home: np.ndarray
    gains: np.ndarray
    lags: np.ndarray
    error: float
    samples: int
    pitch_samples: int

    @property
    def ready(self):
        return centers(self.home, self.geometry)

    def stats(self):
        return dict(ready=self.ready.tolist(), geometry=self.geometry.tolist(), home_angles=self.home.tolist(),
                    angle_channels=self.scripts.tolist(), gain=self.gains.tolist(), lags=self.lags.tolist(),
                    median_fit_error=self.error, motor_samples=self.samples, pitch_samples=self.pitch_samples,
                    rank=3)


def fit_arm(histories, tracker_factory):
    """Fit camera centroids and optional blade pitch against the same paid probes."""
    observations = _body_observations(histories, tracker_factory)
    length = max(map(len, histories))
    commands = np.zeros((len(histories), length, 4))
    point_rows, pitch_rows, episodes, ticks = [], [], [], []
    for episode, (history, sightings) in enumerate(zip(histories, observations, strict=True)):
        commands[episode, :len(history)] = [action for action, _ in history]
        for tick, (point, pitch) in enumerate(sightings):
            if point is not None:
                point_rows.append(point)
                pitch_rows.append(np.nan if pitch is None else pitch)
                episodes.append(episode)
                ticks.append(tick)
    if len(point_rows) < 40:
        raise ValueError('insufficient RGB observations of the controlled blade')
    indices = np.array(episodes), np.array(ticks)
    points, pitches = np.asarray(point_rows), np.asarray(pitch_rows)
    has_pitch = np.isfinite(pitches)
    use_pitch = (has_pitch.sum() >= 35 and np.std(pitches[has_pitch]) > .08)
    geometry = np.array([*MOUNT, LINK])
    position_angles = angles(points, geometry)
    measured = np.column_stack((position_angles, pitches)) if use_pitch else position_angles
    assignments = np.full((measured.shape[1], 4), np.inf)
    lags = np.zeros_like(assignments)
    models = np.zeros((*assignments.shape, 2))
    for lag in (.10, .25, .40, .55, .68, .78, .86):
        response = _filtered(commands, lag)[indices]
        for axis, target in enumerate(measured.T):
            valid = np.isfinite(target)
            scale = max(.02, np.std(target[valid])**2)
            for channel in range(4):
                model, cost = _linear_fit(response[valid, channel], target[valid])
                cost /= scale
                if cost < assignments[axis, channel]:
                    assignments[axis, channel], lags[axis, channel], models[axis, channel] = cost, lag, model
    rows, channels = linear_sum_assignment(assignments)
    if np.max(assignments[rows[:3], channels[:3]]) > .60:
        raise ValueError('probe observations did not identify three independent position axes')
    selected_lags = lags[rows, channels]
    selected_models = models[rows, channels]
    inputs = np.column_stack([_filtered(commands[..., channel], lag)[indices]
                              for channel, lag in zip(channels[:3], selected_lags[:3], strict=True)])

    def residual(parameters):
        body, home, gain = parameters[:4], parameters[4:7], parameters[7:10]
        pose = centers(home + inputs * gain, body)
        calibration = (body - geometry) * np.array([2., 3., 3., 4.])
        return np.concatenate(((pose - points).ravel(), calibration))

    fitted = least_squares(residual, np.concatenate((geometry, selected_models[:3, 0], selected_models[:3, 1])),
                           loss='soft_l1', f_scale=.015, max_nfev=40)
    body, home, gain = fitted.x[:4], fitted.x[4:7], fitted.x[7:10]
    errors = np.linalg.norm(centers(home + inputs * gain, body) - points, axis=1)
    if np.max(np.abs(body - geometry)) > .11 or np.min(np.abs(gain)) < .035:
        raise ValueError('probe solution has implausible link dimensions or actuator response')
    if use_pitch:
        home = np.append(home, selected_models[3, 0])
        gain = np.append(gain, selected_models[3, 1])
    else:
        channels = np.append(channels, next(index for index in range(4) if index not in channels))
        home = np.append(home, HOME_PITCH)
        # Centroid-only rigs cannot identify the direction of the remaining wrist axis.
        gain = np.append(gain, 0.)
        selected_lags = np.append(selected_lags, .45)
    return FittedArm(body, channels, home, gain, selected_lags, float(np.median(errors)), len(points),
                     int(has_pitch.sum()))


class BodyServo:
    """Bound motor targets, predict servo lag, and correct it from the visible pose."""

    def __init__(self, fit):
        self.fit = fit
        self.requested = fit.home.copy()
        self.effective = fit.home.copy()
        self.correction = np.zeros(4)
        self.pose = fit.ready.copy()
        self.velocity = np.zeros(3)
        self.pitch = fit.home[3]

    def advance(self, preceding):
        if preceding is not None:
            increment = self.fit.gains * np.asarray(preceding)[self.fit.scripts]
            self.requested = np.clip(self.requested + increment, ANGLE_LOW, ANGLE_HIGH)
            self.effective = self.fit.lags * self.effective + (1. - self.fit.lags) * self.requested

    def observe(self, actor, pitch):
        expected = centers(self.effective, self.fit.geometry)
        good = actor is not None and actor.observed and np.linalg.norm(np.asarray(actor.position) - expected) < .15
        actual = np.asarray(actor.position) if good else centers(self.effective + self.correction, self.fit.geometry)
        self.velocity = .35 * self.velocity + .65 * (actual - self.pose)
        self.pose = actual
        if good:
            observed = angles(actual, self.fit.geometry)
            self.correction[:3] = .55 * self.correction[:3] + .45 * np.clip(observed - self.effective[:3], -.24, .24)
        if pitch is not None and np.isfinite(pitch) and abs(pitch - self.effective[3]) < .45:
            self.correction[3] = .55 * self.correction[3] + .45 * (pitch - self.effective[3])
        self.pitch = self.effective[3] + self.correction[3]

    def command(self, goal, lead_goal, pitch):
        theta = angles(goal, self.fit.geometry)
        future = angles(lead_goal, self.fit.geometry)
        pitch = np.clip(pitch, ANGLE_LOW[3], ANGLE_HIGH[3])
        # One-frame-ahead position, plus feed-forward for a moving setpoint.
        target = np.append(future + self.fit.lags[:3] * (future - theta) / (1. - self.fit.lags[:3]), pitch)
        corrected = np.clip(target - self.correction, ANGLE_LOW, ANGLE_HIGH)
        increment = corrected - self.requested
        rates = np.divide(increment, self.fit.gains, out=np.zeros(4), where=self.fit.gains != 0.)
        command = np.zeros(4)
        command[self.fit.scripts] = np.clip(rates, -1., 1.)
        return command
