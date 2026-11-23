"""Bind retinal regions to moving and self-related roles without a fixed palette."""

from dataclasses import dataclass
from collections import deque

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .tt24_motor import Sighting


_HUE_BINS = 96
_CIRCLE = 2. * np.pi
_BIN_ANGLE = _CIRCLE / _HUE_BINS


def _angular_distance(a, b):
    return abs((a - b + np.pi) % _CIRCLE - np.pi)


@dataclass(frozen=True)
class _Region:
    sighting: Sighting
    hue: float
    pixels: int
    quality: float
    outline: np.ndarray


def _regions(image):
    """Find connected regions around color modes measured in this frame."""
    rgb = np.asarray(image, dtype=np.float32)
    height, width = rgb.shape[:2]
    red, green, blue = np.moveaxis(rgb, -1, 0)
    chroma_x = 2. * red - green - blue
    chroma_y = np.sqrt(3.) * (green - blue)
    angle = np.arctan2(chroma_y, chroma_x)
    bins = np.rint(angle / _BIN_ANGLE).astype(np.int16) % _HUE_BINS
    colored = np.hypot(chroma_x, chroma_y) > 14. + .07 * (red + green + blue)
    histogram = np.bincount(bins[colored], minlength=_HUE_BINS)
    smooth = 2 * histogram + np.roll(histogram, 1) + np.roll(histogram, -1)
    local_max = (smooth >= np.roll(smooth, 1)) & (smooth >= np.roll(smooth, -1))
    minimum = max(3, round(height * width / 6000))
    maxima = np.flatnonzero(local_max & (smooth >= 2 * minimum))
    peaks = []
    for index in sorted(maxima, key=lambda i: smooth[i], reverse=True):
        if all(min((index - peak) % _HUE_BINS, (peak - index) % _HUE_BINS) > 3 for peak in peaks):
            peaks.append(index)
    if not peaks:
        return []

    differences = abs(np.arange(_HUE_BINS)[:, None] - peaks)
    distances = np.minimum(differences, _HUE_BINS - differences)
    palette = distances.argmin(axis=1) + 1
    palette[distances.min(axis=1) > 7] = 0
    groups = np.where(colored, palette[bins], 0)

    # Local image contrast and edge continuity rate the evidence for a region.
    edge = np.zeros((height, width), np.float32)
    for axis in (0, 1):
        difference = np.linalg.norm(np.diff(rgb, axis=axis), axis=-1)
        if axis == 0:
            np.maximum(edge[1:], difference, out=edge[1:])
            np.maximum(edge[:-1], difference, out=edge[:-1])
        else:
            np.maximum(edge[:, 1:], difference, out=edge[:, 1:])
            np.maximum(edge[:, :-1], difference, out=edge[:, :-1])

    regions = []
    for group in range(1, len(peaks) + 1):
        labels, _ = ndimage.label(groups == group)
        sizes = np.bincount(labels.ravel())
        for index, box in enumerate(ndimage.find_objects(labels), 1):
            size = sizes[index]
            if size < minimum or size > .55 * height * width:
                continue
            y, x = box
            mask = labels[box] == index
            ys, xs = np.nonzero(mask)
            w, h = x.stop - x.start, y.stop - y.start
            center = (np.array((x.start + xs.mean(), y.start + ys.mean())) -
                      ((width - 1) / 2, (height - 1) / 2)) / height
            sighting = Sighting(center, float(np.sqrt(size / np.pi) / height), max(w, h) / (2. * height))
            pixel_angles = angle[box][mask]
            phase = np.mean(np.exp(1j * pixel_angles))
            outline = mask & ~ndimage.binary_erosion(mask)
            contrast = float(np.mean(edge[box][outline])) / 255.
            fill = size / (w * h)
            borders = (x.start == 0 and x.stop == width) or (y.start == 0 and y.stop == height)
            quality = (contrast * np.sqrt(fill) * abs(phase) ** 6 *
                       min(1., np.sqrt(size) / (2. * np.sqrt(minimum))))
            if borders:
                quality *= .35
            out_y, out_x = np.nonzero(outline)
            contour = (np.stack((x.start + out_x, y.start + out_y), axis=1) -
                       ((width - 1) / 2, (height - 1) / 2)) / height
            regions.append(_Region(sighting, float(np.angle(phase)), int(size), float(quality), contour))
    return regions


class _Track:
    def __init__(self, region, action_size):
        self.region = region
        self.velocity = np.zeros(2)
        self.coefficients = np.zeros((action_size, 2))
        self.precision = np.eye(action_size) * 4.
        self.explained = 0.
        self.motion = 0.
        self.speed = 0.
        self.quality = region.quality
        self.interaction = 0.
        self.approach = 0.
        self.hits = 1
        self.missed = 0

    def update(self, region, displacement, background, action):
        previous = self.region.sighting
        size_change = abs(np.log(region.sighting.radius / previous.radius))
        if self.coefficients.shape[0] != len(action):
            self.coefficients = np.zeros((len(action), 2))
            self.precision = np.eye(len(action)) * 4.
        prediction = action @ self.coefficients
        residual = displacement - prediction
        gain = self.precision @ action
        gain /= .98 + action @ gain
        self.coefficients += np.outer(gain, residual)
        self.precision = (self.precision - np.outer(gain, action @ self.precision)) / .98
        speed = np.linalg.norm(displacement)
        explained = max(0., speed - np.linalg.norm(residual))
        independent = min(np.linalg.norm(displacement - background), np.linalg.norm(residual))
        self.velocity = .55 * self.velocity + .45 * displacement
        self.quality = .82 * self.quality + .18 * region.quality * np.exp(-.5 * size_change)
        self.speed = .8 * self.speed + .2 * speed
        self.motion = .78 * self.motion + .22 * independent
        self.explained = .8 * self.explained + .2 * explained
        self.region = region
        self.hits += 1
        self.missed = 0

    def mover_score(self):
        return self.quality * self.motion * min(1., self.hits / 4)

    def anchor_score(self, retinal_noise):
        # Self-related evidence includes either a fixed retinal location or a
        # displacement the observer's actions can explain.
        instability = max(0., self.speed - self.explained)
        support = np.sqrt(self.region.pixels) * min(1., self.hits / 4)
        association = 1. + 1.5 * self.approach + 7. * self.interaction
        return self.quality * support * association / (1. + instability / retinal_noise)


def _approach(region, center, velocity, resolution, scale):
    relative = region.sighting.center - center
    alignment = float(relative @ velocity) / max(resolution**2, np.linalg.norm(relative) * np.linalg.norm(velocity))
    if alignment <= 0.:
        return 0.
    distances = np.linalg.norm(region.outline - center, axis=1)
    gap = max(0., distances.min() - scale)
    return alignment**2 * np.exp(-gap / max(2. * resolution, scale))


class AnonymousVision:
    """Retain anonymous appearances, trajectories, and local action effects."""

    def __init__(self):
        self.tracks = []
        self.mover = None
        self.anchor = None
        self.history = deque(maxlen=3)
        self.drive = None

    def observe(self, rgb, previous_action=None):
        """Read mover and retinal anchor moments from RGB and recent own action."""
        if rgb is None:
            return None, None
        action = np.asarray(previous_action, dtype=float) if previous_action is not None else None
        if action is not None:
            if self.drive is None:
                self.drive = np.zeros_like(action)
            self.drive = .35 * self.drive + .65 * action
        drive = self.drive if self.drive is not None else np.empty(0)

        regions = _regions(rgb)
        matched = self._assign(regions)
        # A weighted median removes motion shared by the larger retinal regions.
        # The cap limits the influence of any single region crossing the frame.
        background = np.zeros(2)
        if len(matched) >= 3:
            weights = np.array([min(self.tracks[i].region.pixels, rgb.shape[0] * rgb.shape[1] / 12.)
                                for i, _, _ in matched])
            motion = np.array([delta for _, _, delta in matched])
            for axis in range(2):
                order = np.argsort(motion[:, axis])
                middle = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)
                background[axis] = motion[order[middle], axis]

        for track in self.tracks:
            track.missed += 1
        seen = set()
        for i, j, delta in matched:
            self.tracks[i].update(regions[j], delta, background, drive)
            seen.add(j)
        self.tracks = [track for track in self.tracks if track.missed <= 4]
        self.tracks.extend(_Track(region, len(drive)) for j, region in enumerate(regions) if j not in seen)

        noise = .55 / rgb.shape[0]
        measured = [track for track in self.tracks if not track.missed and track.hits >= 2]
        plausible = [track for track in measured if track.motion >= .55 * noise or track is self.mover]
        mover = max(plausible, key=lambda t: t.mover_score() * (1.5 if t is self.mover else 1.), default=None)
        if (mover is not None and self.mover is not None and 0 < self.mover.missed <= 3 and
                mover.mover_score() < self.mover.mover_score()):
            mover = None
        if mover is None or mover is not self.mover:
            self.history.clear()
        if mover is not None:
            self.mover = mover
            self.history.append(mover.region.sighting)
        anchors = [t for t in measured if t is not self.mover]
        self._update_relations(anchors, noise)
        anchor = max(anchors,
                     key=lambda t: t.anchor_score(noise) * (1.12 if t is self.anchor else 1.), default=None)
        self.anchor = anchor
        return (None if mover is None else mover.region.sighting,
                None if anchor is None else anchor.region.sighting)

    def _update_relations(self, candidates, noise):
        for track in candidates:
            track.interaction *= .985
            track.approach = 0.
        if len(self.history) < 2:
            return
        previous, latest = self.history[-2], self.history[-1]
        outgoing = latest.center - previous.center
        shock = 0.
        if len(self.history) == self.history.maxlen:
            incoming = previous.center - self.history[0].center
            if incoming @ outgoing <= 0.:
                shock = np.clip(np.linalg.norm(outgoing - incoming) / (3. * noise) - 1., 0., 1.)
        for track in candidates:
            track.approach = _approach(track.region, latest.center + outgoing, outgoing, noise, latest.radius)
            if shock:
                contact = _approach(track.region, previous.center, incoming, noise, previous.radius)
                track.interaction = max(track.interaction, shock * contact)

    def _assign(self, regions):
        if not self.tracks or not regions:
            return []
        region_centers = np.stack([region.sighting.center for region in regions])
        hues = np.array([region.hue for region in regions])
        radii = np.array([region.sighting.radius for region in regions])
        cost = np.empty((len(self.tracks), len(regions)))
        for index, track in enumerate(self.tracks):
            sighting = track.region.sighting
            predicted = sighting.center + (track.missed + 1) * track.velocity
            displacement = np.linalg.norm(region_centers - predicted, axis=1)
            angular = _angular_distance(hues, track.region.hue)
            tolerance = .065 * (track.missed + 1) + sighting.radius + radii
            scale = abs(np.log(radii / sighting.radius))
            cost[index] = np.where((displacement < tolerance) & (angular < .34) & (scale < 1.2),
                                   displacement / tolerance + angular / .34 + .3 * scale, 10.)
        rows, columns = linear_sum_assignment(cost)
        return [(i, j, (region_centers[j] - self.tracks[i].region.sighting.center) / (self.tracks[i].missed + 1))
                for i, j in zip(rows, columns, strict=True) if cost[i, j] < 10.]
