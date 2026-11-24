"""Causal far-racket measurements in the unchanged, calibrated stereo RGB."""

from collections import deque
from itertools import product

import numpy as np
from scipy.ndimage import find_objects, label


def _bounds(camera):
    points = np.array(list(product((-.93, .93), (.24, 1.98), (.65, 1.74))))
    xy = camera.project(points)[:, :2]
    lo = np.maximum(0, np.floor(xy.min(axis=0)).astype(int) - 9)
    hi = np.minimum((camera.width, camera.height), np.ceil(xy.max(axis=0)).astype(int) + 9)
    return (*lo, *hi)


def _blobs(rgb, bounds):
    x0, y0, x1, y1 = bounds
    r, g, b = np.moveaxis(rgb[y0:y1, x0:x1].astype(np.int16), -1, 0)
    # Gray rubber in the existing natural palette. No markers, no render IDs.
    mask = ((26 < r) & (r < 110) & (0 < g - r) & (g - r <= 8)
            & (2 <= b - g) & (b - g <= 10))
    regions, _ = label(mask, structure=np.ones((3, 3)))
    for index, (ys, xs) in enumerate(find_objects(regions), 1):
        w, h = xs.stop - xs.start, ys.stop - ys.start
        if not 4 <= min(w, h) <= max(w, h) <= 30:
            continue
        y, x = np.nonzero(regions[ys, xs] == index)
        n = len(x)
        if n < 15 or n < .31 * w * h:
            continue
        uv = np.c_[x + xs.start + x0, y + ys.start + y0]
        center = uv.mean(axis=0)
        d = uv - center
        moment = d.T @ d / n
        shape = np.array([moment[0, 0], moment[1, 1], moment[0, 1]])
        yield center, shape, n


class PaddleEye:
    """Track center, velocity and each view's silhouette moments, without opponent state."""

    def __init__(self, views):
        self.views = views
        self.crop = {name: _bounds(view) for name, view in views.items()}
        self.samples = deque(maxlen=5)

    def update(self, now, frames):
        if frames is None or not all(name in frames for name in self.views):
            return None
        if self.samples and now - self.samples[-1][0] > .15:
            self.samples.clear()
        left, right = tuple(self.views.values())
        candidates = [_blobs(frames[name], self.crop[name]) for name in self.views]
        best = None
        for (u, a, n), (v, b, m) in product(*candidates):
            r, s = left.ray(u), right.ray(v)
            basis = np.c_[r, -s]
            first, second = np.linalg.lstsq(basis, right.eye - left.eye, rcond=None)[0]
            p, q = left.eye + first * r, right.eye + second * s
            center = .5 * (p + q)
            gap = np.linalg.norm(p - q)
            x, y, z = center
            if (gap > .027 or abs(x) >= .95 or not .23 < y < 2.02
                    or not .59 < z < 1.80):
                continue
            # Downweight stray rails, shadows or cases with unresolvable silhouettes.
            ratio = n * first**2 / left.focal**2 / (m * second**2 / right.focal**2)
            if not .45 < ratio < 2.22:
                continue
            continuity = 0. if not self.samples else np.linalg.norm(center - self.samples[-1][1][:3])
            cost = gap * 32. + abs(np.log(ratio)) + .8 * continuity - .004 * min(n + m, 250)
            shape = np.r_[a * (first / left.focal) ** 2, b * (second / right.focal) ** 2]
            if best is None or cost < best[0]:
                best = cost, np.r_[center, shape]
        if best is None:
            return None
        self.samples.append((now, best[1]))
        if len(self.samples) < 3:
            return None
        times = np.array([t for t, _ in self.samples])
        values = np.array([x for _, x in self.samples])
        if not .045 < now - times[0] <= .18:
            return None
        dt = times - times.mean()
        velocity = dt @ values[:, :3] / (dt @ dt)
        if np.linalg.norm(velocity) > 5.2:
            return None
        return np.r_[values[-1, :3], velocity, np.mean(values[-3:, 3:], axis=0)]


def quality(f, now):
    return (f is not None and abs(now - f.time) <= 1e-7 and f.frames >= 3
            and f.rms < 2.40 and f.depth_sigma < .16)


class CueGate:
    """Use the same single outbound ball gate in every arm of the comparison."""

    def __init__(self, line=1.0):
        self.line = line
        self.latched = False

    def update(self, flight, now):
        if not quality(flight, now):
            return None
        if flight.velocity[1] < -1.:
            self.latched = False
        if self.latched or flight.velocity[1] < 1. or not self.line <= flight.position[1] < 1.82:
            return None
        self.latched = True
        return np.r_[flight.position, flight.velocity]


def plane_sample(flight, now, line=-.40):
    """Trusted later optical p/v, re-timed to a fixed plane with only a small interpolation."""
    if not quality(flight, now) or flight.velocity[1] > -1.35:
        return None
    travel = (line - flight.position[1]) / flight.velocity[1]
    if abs(travel) >= .100:
        return None
    damp = flight.damping
    delta = -np.log1p(-damp * travel) / damp
    p, v = flight.advance(now + delta)
    return abs(delta), now + delta, np.r_[p[[0, 2]], v]
