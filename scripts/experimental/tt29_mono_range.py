"""Learn metric apparent ball size from paired RGB and deploy one fixed camera."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .tt29_vision import OpticalFlight, RaySample, ball_pixels


@dataclass(frozen=True)
class Appearance:
    features: np.ndarray
    hard: RaySample


def appearance(image, camera, sight, time):
    """Collect local radial color flux, effective size and quantization features."""
    hard = RaySample.measure(time, "left", camera, sight, 1.)
    x, y = sight.center
    half = int(np.clip(np.ceil(sight.radius * 1.6 + 2.5), 5, 15))
    xi, yi = round(x), round(y)
    if sight.clipped or xi < half or yi < half or xi + half >= camera.width or yi + half >= camera.height:
        return None
    patch = image[yi-half:yi+half+1, xi-half:xi+half+1].astype(float)
    dy, dx = np.mgrid[-half:half+1, -half:half+1]
    squared = (dx + xi - x)**2 + (dy + yi - y)**2
    r, g, b = np.moveaxis(patch, -1, 0)
    # Ball color has strong red excess and moderate green. The softer hue
    # support includes anti-aliased boundaries too faint for the hard mask.
    chroma = r - b
    green = (g - b) / np.maximum(1., chroma)
    hue = np.clip(5 * (green - .075), 0., 1.) * np.clip(5 * (.76 - green), 0., 1.)
    surround = np.median(chroma[squared > (half - 1)**2])
    support = np.maximum(0., chroma - max(10., surround + 8.)) * hue
    support[squared > (1.65 * sight.radius + 1.75)**2] = 0.
    flux = np.sum(support)
    peak = np.max(support)
    effective = flux / max(1., peak)
    spread = np.sqrt(np.sum(support * squared) / max(1., flux))
    ratio = np.log(max(.035, effective / sight.area))
    normalized_spread = spread / sight.radius - .70
    phase = (sight.center - np.round(sight.center))
    features = np.array([1., np.log(sight.radius / 3.), ratio,
                         normalized_spread, (peak - 120.) / 100.,
                         (effective / max(1, sight.area))**2,
                         phase @ phase - .10])
    return Appearance(features, hard)


def triangulated_range(left, right):
    dot = left.direction @ right.direction
    if abs(dot) >= .97:
        return None
    translation = right.eye - left.eye
    l, r = translation @ left.direction, translation @ right.direction
    first, second = (l - dot * r) / (1 - dot * dot), (dot * l - r) / (1 - dot * dot)
    gap = np.linalg.norm(left.eye + first * left.direction - right.eye - second * right.direction)
    if .3 < first < 8 and .3 < second < 8 and gap < .016:
        return float(first)
    return None


class RangeFit:
    """Fit a low-dimensional correction to the analytic angular-size range."""

    def __init__(self):
        self.matrix = np.diag([.5, 1., 1.5, 2.5, 3., 3., 3.])
        self.correlation = np.zeros(7)
        self.examples = []
        self.coefficients = np.zeros(7)
        self.sigma = .12

    def add(self, sight, distance):
        features = sight.features
        target = np.log(distance / sight.hard.range_m)
        if abs(target) > .52:
            return False
        self.matrix += np.outer(features, features)
        self.correlation += features * target
        self.examples.append((features.copy(), sight.hard.range_m, distance))
        self.coefficients = np.linalg.solve(self.matrix, self.correlation)
        return True

    def freeze(self):
        features, initial, labels = zip(*self.examples, strict=True)
        error = np.asarray(initial) * np.exp(np.asarray(features) @ self.coefficients) - labels
        self.sigma = float(np.clip(1.4826 * np.median(abs(error - np.median(error))), .045, .35))
        return self

    def range(self, sight):
        correction = float(np.clip(sight.features @ self.coefficients, -.36, .45))
        return float(sight.hard.range_m * np.exp(correction))

    def receipt(self):
        return dict(coefficients=self.coefficients.tolist(), robust_range_sigma_m=self.sigma,
                    sample_count=len(self.examples), gram=self.matrix.tolist(), correlation=self.correlation.tolist())

    @classmethod
    def restore(cls, receipt):
        model = cls()
        model.coefficients = np.array(receipt["coefficients"])
        model.sigma = float(receipt["robust_range_sigma_m"])
        return model


class MonoFlight(OpticalFlight):
    """A frozen optical correction augments the existing causal drag/gravity fit."""

    def __init__(self, rig, model):
        super().__init__(rig)
        self.model = model

    def observe(self, capture, frames, encoders):
        if self.last_capture is not None and capture <= self.last_capture:
            raise ValueError("camera captures must be passed once, in timestamp order")
        self.last_capture = capture
        image = frames["left"]
        camera = self.rig.fixed["left"]
        predicted = None
        if self.latest is not None and capture - self.latest.time < .08:
            last = camera.project(self.latest.advance(capture)[0])
            if last[2] > .05:
                predicted = last[:2]
        seen = ball_pixels(image, predicted)
        self.last_frame_views = ("left",) if seen is not None else ()
        if seen is None:
            return self.latest
        shape = appearance(image, camera, seen, capture)
        if shape is None:
            sample = RaySample.measure(capture, "left", camera, seen, 1.)
        else:
            distance = self.model.range(shape)
            hard = shape.hard
            # Use conservatively calibrated size uncertainty relative to the
            # angular measurement, with the baseline zero-weight on single pixels.
            weight = min(5.0, 1. / (3.7 * self.model.sigma * (1 + .12 * distance))) if hard.depth_weight else 0.
            sample = RaySample(capture, "left", hard.eye, hard.direction, distance, weight,
                               camera.focal / max(.30, distance))
        self.rays.append(sample)
        while self.rays[0].time < capture - self.window * 1.35:
            self.rays.popleft()
        self._fit(capture)
        return self.latest
