"""Depth priors for an RGB ball seen in a single registered camera."""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RayEstimate:
    position: np.ndarray
    range_half_width: float
    source_frames: int


@dataclass(frozen=True)
class Ray:
    eye: np.ndarray
    direction: np.ndarray

    @classmethod
    def observed(cls, camera, pixel):
        direction = camera.ray(pixel)
        return cls(camera.eye, direction / np.linalg.norm(direction))

    def distance(self, point):
        return float((point - self.eye) @ self.direction)

    def point(self, distance):
        return self.eye + distance * self.direction


def _apparent_size(camera, blob):
    # Chroma-weighted image area, normalized for lens focal length and the
    # increase in projected area toward the image edge. The scale itself is
    # learned from this stream's stereo history.
    area = blob.weights.sum() / 255.
    obliquity = np.linalg.norm(camera.ray(blob.center))
    return np.sqrt(area / obliquity) / camera.focal


class AppearanceDepth:
    """Learn a stream's apparent scale using only earlier RGB triangulations."""

    def __init__(self):
        self.streams = {}
        self.source_frames = 0

    def observe(self, fits, blobs, position):
        if position is None or len(fits) < 2:
            return
        observations = {}
        for stream in fits.keys() & blobs.keys():
            camera, blob = fits[stream], blobs[stream]
            u, v, depth = camera.project(position)
            if depth > 0 and np.linalg.norm((u, v) - blob.center) <= 1.5:
                observations[stream] = (camera, float(depth), _apparent_size(camera, blob))
        if len(observations) < 2:
            return
        self.source_frames += 1
        for stream, (camera, depth, size) in observations.items():
            prior = self.streams.get(stream)
            if prior is None or prior[0] is not camera:
                prior = self.streams[stream] = (camera, deque(maxlen=256))
            prior[1].append((depth, size))

    def predict(self, stream, camera, blob):
        """Return a size prediction, an estimated range width and a median-depth control."""
        entry = self.streams.get(stream)
        if entry is None or entry[0] is not camera:
            return None, None
        observations = entry[1]
        depth, appearances = np.asarray(observations).T
        ray = camera.ray(blob.center)
        control = camera.eye + np.median(depth) * ray
        if len(depth) < 8:
            return None, control
        size = _apparent_size(camera, blob)
        scale = np.log(depth * appearances)
        typical = np.median(scale)
        predicted_depth = np.exp(typical) / size
        scatter = np.quantile(np.abs(scale - typical), .90)
        # One rasterized pixel makes a larger fractional difference for a small
        # blob. A repeatable fit on quantized source images cannot remove this.
        log_width = max(scatter, .5 / len(blob.pixels))
        range_width = np.linalg.norm(ray) * predicted_depth * np.expm1(log_width)
        return RayEstimate(camera.eye + predicted_depth * ray, float(range_width), len(depth)), control

    def qualified(self, stream, estimate, camera, blob, *, max_width=.15):
        """Require supported image size, depth variation and a useful range width."""
        if estimate is None or estimate.range_half_width > max_width:
            return False
        depth, size = np.asarray(self.streams[stream][1]).T
        near, far = np.quantile(depth, (.1, .9))
        return bool(far > 1.10 * near and min(size) <= _apparent_size(camera, blob) <= max(size))


class StereoContinuation:
    """Carry coherent stereo motion briefly through a loss of one camera."""

    def __init__(self, frame_seconds):
        self.frame_seconds = frame_seconds
        self.stereo = deque(maxlen=4)

    def observe(self, tick, position):
        if position is not None:
            self.stereo.append((tick, position))

    def predict(self, tick, camera, blob):
        if len(self.stereo) < self.stereo.maxlen:
            return None, {}
        ticks, positions = zip(*self.stereo, strict=True)
        if np.any(np.diff(ticks) != 1) or not 1 <= tick - ticks[-1] <= 4:
            return None, {}
        times = (np.asarray(ticks) - ticks[-1]) * self.frame_seconds
        horizon = (tick - ticks[-1]) * self.frame_seconds
        gravity = np.array((0., 0., -9.81))
        design = np.column_stack((np.ones(len(times)), times))
        positions = np.asarray(positions)
        ballistic = positions - .5 * times[:, None]**2 * gravity
        offset, velocity = np.linalg.lstsq(design, ballistic, rcond=None)[0]
        current = offset + horizon * velocity + .5 * horizon**2 * gravity
        ray = Ray.observed(camera, blob.center)
        distance = ray.distance(current)
        constrained = ray.point(distance)
        linear = np.array((1., horizon)) @ np.linalg.lstsq(design, positions, rcond=None)[0]
        controls = {'last_point': positions[-1], 'linear': linear,
                    'linear_on_ray': ray.point(ray.distance(linear)), 'gravity': current}
        residuals = np.linalg.norm(design @ np.stack((offset, velocity)) - ballistic, axis=1)
        # Include source measurement scatter, extrapolation growth and a slack
        # term for unmodeled acceleration. A contact can invalidate the prior.
        scatter = max(.012, float(np.max(residuals)))
        half_width = 2. * scatter * (1. + horizon / abs(times[0])) + 3. * horizon**2
        lateral_error = np.linalg.norm(current - constrained)
        if distance <= 0. or lateral_error > half_width or half_width > .15:
            return None, controls
        return RayEstimate(constrained, float(half_width), len(ticks)), controls
