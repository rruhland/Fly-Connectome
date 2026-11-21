"""Recover a metric ball arc from successive bearings in a single RGB stream."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .tt4_arena import BALL_RADIUS, GRAVITY


@dataclass(frozen=True)
class _Sighting:
    tick: int
    eye: np.ndarray
    axes: np.ndarray
    bearing_rows: np.ndarray
    focal: float
    pixel_sigma: float
    depth: float
    size_sigma: float


class MonoFlight:
    """Fit the latest arc using only time, pixels, camera calibration and court geometry.

    A separate instance processes each camera stream per point. Each successful
    observation reports the table-relative position and per-axis uncertainty in
    metres. The corresponding velocity is available in metres per camera frame.
    """

    def __init__(self, *, bounds=(.424, .80), frame_seconds=1 / 30, law=None):
        self.bounds = np.asarray(bounds)
        self.frame_seconds = frame_seconds
        self.gravity = GRAVITY * frame_seconds**2 if law is None else law.gravity * (30 * frame_seconds)**2
        # The linear drag approximation uses a representative table-tennis speed.
        self.drag = (.17 if law is None else law.drag) * 2.8 * frame_seconds
        self.tangent = .945 if law is None else law.tangent
        self.restitution = .86 if law is None else law.restitution
        self.sightings = deque(maxlen=14)
        self.reset()

    def reset(self):
        self.sightings.clear()
        self.velocity = self.source = self.bounced = None

    def _terms(self, time):
        if self.drag == 0.:
            return np.ones_like(time), time, .5 * self.gravity * time**2
        decay = np.exp(-self.drag * time)
        move = -np.expm1(-self.drag * time) / self.drag
        drop = (time - move) * self.gravity / self.drag
        return decay, move, drop

    def _samples(self, points):
        eyes = np.stack([p.eye for p in points])
        axes = np.stack([p.axes for p in points])
        bearing = np.stack([p.bearing_rows for p in points])
        focals = np.array([p.focal for p in points])
        pixel_sigma = np.array([p.pixel_sigma for p in points])
        size_depth = np.array([p.depth for p in points])
        size_sigma = np.array([p.size_sigma for p in points])
        hint = np.where(size_depth > 0., size_depth, np.clip(np.linalg.norm(eyes, axis=1), .2, 3.))
        return eyes, axes, bearing, focals, pixel_sigma, size_depth, size_sigma, hint

    def _fit(self, basis, offset, samples):
        eyes, axes, bearing, focals, pixel_sigma, size_depth, size_sigma, hint = samples
        screen = bearing * (focals / hint / pixel_sigma)[:, None, None]
        scale = axes[:, -1] / size_sigma[:, None]
        rows = np.concatenate((screen, scale[:, None, :]), axis=1)
        design = np.einsum('nij,cnjd->cnid', rows, basis)
        target = np.einsum('nij,cnj->cni', rows, eyes - offset)
        target[..., 2] += size_depth / size_sigma
        matrix = np.einsum('cnid,cnie->cde', design, design)
        rest = np.einsum('cnid,cni->cd', design, target)
        matrix += 1e-8 * np.eye(matrix.shape[-1])
        estimate = np.linalg.solve(matrix, rest[..., None])[..., 0]
        path = np.einsum('cnid,cd->cni', basis, estimate) + offset
        eye_delta = path - eyes
        depth = np.einsum('ni,cni->cn', axes[:, -1], eye_delta)
        pixel_error = (np.einsum('nij,cnj->cni', bearing, eye_delta)
                       * (focals / np.maximum(depth, .05))[..., None])
        error = np.linalg.norm(pixel_error, axis=-1)
        error_weighted = error / pixel_sigma
        size_error = (depth - size_depth) / size_sigma
        costs = np.mean(error_weighted**2, axis=1) + .22 * np.mean(size_error**2, axis=1)
        over_court = (np.abs(path[..., :2]) <= self.bounds + .013).all(axis=-1)
        qualified = ((depth > .06).all(axis=1) & (np.abs(path[..., 0]) < .99).all(axis=1)
                     & (np.abs(path[..., 1]) < max(1.95, self.bounds[1] + .60)).all(axis=1)
                     & (path[..., 2] < 1.25).all(axis=1) & (path[:, -1, 2] > -.16)
                     & ~np.any(over_court & (path[..., 2] < -.019), axis=1)
                     & (np.max(error_weighted, axis=1) < 3.8)
                     & (np.mean(error_weighted**2, axis=1) < 3.))
        costs[~qualified] = np.inf
        return path, estimate, matrix, costs

    def _smooth(self, t, samples):
        decay, move, drop = self._terms(t)
        time_scale = max(3., -t[0])
        eye = np.eye(3)
        basis = np.empty((1, len(t), 3, 6))
        basis[0, ..., :3] = eye
        basis[0, ..., 3:] = move[:, None, None] / time_scale * eye
        offset = np.zeros((1, len(t), 3))
        offset[0, :, 2] = -drop
        path, estimate, matrices, cost = self._fit(basis, offset, samples)
        candidate = (float(cost[0]), path[0, -1], estimate[0, 3:] * decay[-1] / time_scale,
                     basis[0, -1], matrices[0], 'ballistic', False)
        return candidate

    def _bounce(self, t, samples):
        contacts = np.arange(t[0] + 1., t[-1] - .15, .30)
        if contacts.size == 0:
            return None
        time = t - contacts[:, None]
        after = time >= 0.
        decay, move, drop = self._terms(time)
        multipliers = np.where(after[..., None], [self.tangent, self.tangent, -self.restitution], 1.)
        time_scale = max(3., -t[0])
        basis = np.empty((*time.shape, 3, 5))
        basis[..., :2] = np.eye(3)[:, :2]
        basis[..., 2:] = (move[..., None] / time_scale * multipliers)[..., None, :] * np.eye(3)
        offset = np.zeros((*time.shape, 3))
        offset[..., 2] = BALL_RADIUS - drop
        paths, estimates, matrices, costs = self._fit(basis, offset, samples)
        before_velocity = estimates[:, 2:] / time_scale
        bounds = (np.abs(estimates[:, :2]) <= self.bounds + .012).all(axis=1)
        costs[~bounds | (before_velocity[:, 2] > -.017)] = np.inf
        pick = int(np.argmin(costs))
        velocity = (before_velocity[pick] * multipliers[pick, -1] * decay[pick, -1]
                    - np.array((0., 0., self.gravity * move[pick, -1])))
        return (float(costs[pick]) + .18, paths[pick, -1], velocity,
                basis[pick, -1], matrices[pick], 'ballistic_bounce', estimates[pick, 1] < 0.)

    def _solve(self):
        points = list(self.sightings)
        candidates = []
        for length in sorted({len(points), min(8, len(points)), min(5, len(points))}, reverse=True):
            recent = points[-length:]
            t = np.array([p.tick - recent[-1].tick for p in recent])
            samples = self._samples(recent)
            estimate = self._smooth(t, samples)
            candidates.append((estimate[0] + .34 * (len(points) - length), estimate))
            if length >= 6 and estimate[0] > .32:
                bounce = self._bounce(t, samples)
                if bounce is not None:
                    candidates.append((bounce[0] + .34 * (len(points) - length), bounce))
            if estimate[0] <= .32:
                break

        qualified = [(score, state) for score, state in candidates
                     if np.isfinite(score) and np.linalg.norm(state[2]) <= .30 * self.frame_seconds * 30.]
        if not qualified:
            return None
        _, (cost, position, velocity, projection, precision, source, bounced) = min(qualified, key=lambda row: row[0])
        covariance = projection @ np.linalg.solve(precision, projection.T)
        sigma = np.sqrt(np.maximum(0., np.diag(covariance)) * max(1., cost) + .008**2)
        if sigma.max() > .30:
            return None
        self.velocity, self.source, self.bounced = velocity, source, bool(bounced)
        return np.asarray(position), tuple(sigma.tolist())

    def observe(self, tick, fitted_camera, ball_blob, size_position=None, *, size_sigma=None):
        """Add the current pixel and optional metric estimate from its apparent size."""
        self.velocity = self.source = self.bounced = None
        if fitted_camera is None or ball_blob is None:
            return None
        if self.sightings and not 0 < tick - self.sightings[-1].tick <= 3:
            self.sightings.clear()
        while self.sightings and tick - self.sightings[0].tick > 16:
            self.sightings.popleft()
        camera = fitted_camera
        uv = (ball_blob.center - camera.center) / camera.focal
        bearing = camera.rotation[:2] - uv[:, None] * camera.rotation[2]
        depth, uncertainty = 0., np.inf
        if size_position is not None:
            depth = float((np.asarray(size_position) - camera.eye) @ camera.rotation[2])
            radius = camera.focal * BALL_RADIUS / max(depth, .06)
            uncertainty = depth * (.035 + .48 / radius) if size_sigma is None else float(np.max(size_sigma))
            uncertainty = float(np.clip(uncertainty, .027, .45))
            if not np.isfinite(depth) or depth < .06:
                depth, uncertainty = 0., np.inf
        pixel_error = float(np.clip(.52 + .38 * camera.rms, .52, 1.6))
        self.sightings.append(_Sighting(tick, camera.eye, camera.rotation, bearing, camera.focal,
                                        pixel_error, depth, uncertainty))
        if len(self.sightings) < 4 or (len(self.sightings) < 6 and all(p.depth == 0. for p in self.sightings)):
            return None
        return self._solve()
