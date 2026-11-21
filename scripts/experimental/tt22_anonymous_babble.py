"""Find visual track proposals from symmetric motor probes and grayscale changes."""

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal
from scipy.optimize import linear_sum_assignment

from .tt3_visual import RoleBinder, Track


def gray(rgb):
    return np.mean(rgb, axis=2, dtype=np.float32) / 255.


def components(strength, threshold, closing, minimum):
    foreground = ndimage.binary_closing(strength > threshold, iterations=closing)
    foreground = ndimage.binary_fill_holes(foreground)
    regions, count = ndimage.label(foreground)
    for identity, box in enumerate(ndimage.find_objects(regions, max_label=count), 1):
        section = regions[box] == identity
        y, x = np.nonzero(section)
        if len(x) < minimum:
            continue
        y, x = y + box[0].start, x + box[1].start
        yield np.array((x.mean(), y.mean())), float(len(x)), float(strength[y, x].sum())


@dataclass
class Patch:
    reference: np.ndarray
    point: np.ndarray
    support: float


def patch_bank(reference, response):
    """Build a compact multiscale bank from every actuator's image-space effect."""
    candidates = []
    for axis in response:
        for closing in (1, 4):
            moves = sorted(components(axis, .043, closing, 12), key=lambda row: -row[2])
            candidates.extend((center, area, weight) for center, area, weight in moves[:2])
    active = np.count_nonzero(response > .043, axis=0)
    combined = response.max(axis=0) * active
    for threshold in (.08, .16, .28):
        for center, area, weight in components(combined, threshold, 3, 18):
            candidates.append((center, area, weight))
    bank = []
    height, width = reference.shape
    ranked = sorted(candidates, key=lambda row: -row[2] / np.sqrt(row[1]))
    # A single connected response can span several linked moving surfaces.
    # Sample well-separated local responses as well as whole-region centers.
    peaks = ndimage.gaussian_filter(combined, 3.)
    for _ in range(24):
        y, x = np.unravel_index(np.argmax(peaks), peaks.shape)
        strength = peaks[y, x]
        if strength < .034:
            break
        ranked.append((np.array((x, y)), 156., 50. * strength))
        peaks[max(0, y-9):y+10, max(0, x-9):x+10] = 0.
    for center, area, weight in ranked:
        x, y = np.rint(center).astype(int)
        if any(np.linalg.norm(patch.point - (x, y)) < 6. for patch in bank):
            continue
        radius = int(np.clip(.85 * np.sqrt(area), 9, 23))
        if not radius <= x < width - radius or not radius <= y < height - radius:
            continue
        template = reference[y-radius:y+radius+1, x-radius:x+radius+1].copy()
        if np.std(template) < .016:
            continue
        bank.append(Patch(template, np.array((x, y), dtype=float), weight / np.sqrt(area)))
        if len(bank) == 16:
            break
    return bank


def patch_match(image, patch, last, gap):
    margin = 17 + min(gap, 5) * 4
    radius = len(patch) // 2
    x, y = np.rint(last).astype(int)
    left, right = max(0, x - radius - margin), min(image.shape[1], x + radius + margin + 1)
    top, bottom = max(0, y - radius - margin), min(image.shape[0], y + radius + margin + 1)
    local = image[top:bottom, left:right]
    if min(local.shape) < len(patch):
        return last, 0.
    bias = np.mean(patch)
    centered = patch - bias
    n = len(patch)**2

    def moments(values):
        sums = np.pad(np.cumsum(np.cumsum(values, axis=0), axis=1), ((1, 0), (1, 0)))
        width = len(patch)
        return sums[width:, width:] - sums[:-width, width:] - sums[width:, :-width] + sums[:-width, :-width]

    summed = moments(local)
    variance = np.maximum(1.e-5, moments(local * local) - summed**2 / n)
    similarity = signal.correlate(local, centered, mode='valid', method='fft')
    similarity /= np.sqrt(variance * np.sum(centered**2))
    yy, xx = np.ogrid[top+radius:top+radius+similarity.shape[0], left+radius:left+radius+similarity.shape[1]]
    cost = similarity - .9 * np.abs(summed / n - bias) - .00022 * ((xx - x)**2 + (yy - y)**2)
    row, column = np.unravel_index(np.argmax(cost), cost.shape)
    confidence = similarity[row, column] - .9 * abs(summed[row, column] / n - bias)
    return np.array((left + radius + column, top + radius + row), dtype=float), float(confidence)


@dataclass
class History:
    path: deque
    age: int = 0

    @classmethod
    def fresh(cls, tick, center):
        return cls(deque([(tick, center)], maxlen=5), 1)

    def append(self, tick, position):
        self.path.append((tick, position))
        self.age += 1

    def velocity(self):
        return (self.path[-1][1] - self.path[0][1]) / max(1, self.path[-1][0] - self.path[0][0])


class BabblingEye:
    """A grayscale proposal bank. Source ranking and active ranking see the same bank."""

    def __init__(self, patches, role_model):
        self.patches = patches
        self.patch_paths = [None] * len(patches)
        self.motion = {}
        self.next_region = len(patches)
        self.latest = ()
        self.votes = {}
        self.previous = deque(maxlen=2)
        self.tick = -1
        self.binder = RoleBinder(role_model)
        self.binder._scene.tracker = self

    def update(self, frame):
        return self.latest

    def observe(self, frame, command, *, bind=True):
        self.tick += 1
        level = gray(frame)
        height, width = level.shape
        center = np.array((width, height)) / 2
        scale = max(width, height)
        tracks, votes = [], {}

        def add(identity, history):
            track = Track(identity, tuple((history.path[-1][1] - center) / scale),
                          tuple(history.velocity() / scale), True, history.age)
            tracks.append(track)
            return track

        for index, original in enumerate(self.patches):
            path = self.patch_paths[index]
            gap = self.tick - path.path[-1][0] - 1 if path else 0
            previous = path.path[-1][1] if path else original.point
            point, confidence = patch_match(level, original.reference, previous, gap)
            if confidence < .50:
                continue
            if path is None or gap > 4:
                self.patch_paths[index] = path = History.fresh(self.tick, point)
            else:
                path.append(self.tick, point)
            track = add(index, path)
            votes[index] = (track, confidence * confidence * original.support)

        if self.previous:
            changes = np.abs(level - self.previous[0])
            ranked = sorted(components(changes, .035, 1, 2), key=lambda row: -row[2] / np.sqrt(row[1]))
            centers = np.array([row[0] for row in ranked[:20]])
            if len(centers):
                self._match_regions(centers)
        self.previous.append(level)
        for index, path in self.motion.items():
            if path.path[-1][0] == self.tick:
                add(index, path)
        self.latest = tuple(tracks)
        self.votes = votes
        source = self.binder.update(frame, command) if bind else self.latest
        active = max(votes.values(), key=lambda row: row[1], default=(None, 0.))
        motion = [track for track in self.latest if track.id >= len(self.patches) and track.age >= 5]
        fastest = max(motion, key=lambda track: np.linalg.norm(track.velocity), default=None)
        return source, active[0], fastest

    def _match_regions(self, centers):
        self.motion = {key: path for key, path in self.motion.items() if self.tick - path.path[-1][0] <= 3}
        assigned = set()
        if self.motion:
            active = list(self.motion.values())
            estimate = np.array([path.path[-1][1] + (self.tick - path.path[-1][0]) * path.velocity()
                                 for path in active])
            distance = np.linalg.norm(np.subtract(estimate[:, None], centers), axis=2)
            rows, columns = linear_sum_assignment(distance)
            for row, column in zip(rows, columns, strict=True):
                if distance[row, column] <= 18.:
                    active[row].append(self.tick, centers[column])
                    assigned.add(column)
        for index, position in enumerate(centers):
            if index not in assigned:
                self.motion[self.next_region] = History.fresh(self.tick, position)
                self.next_region += 1
