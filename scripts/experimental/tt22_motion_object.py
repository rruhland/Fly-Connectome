"""Carry the appearance of a short anonymous stereo motion track into live play."""

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .tt18_view import triangulate_views
from .tt20_natural import NaturalViewTransport
from .tt22_motion_prefix import candidates, paths


OPENING = 8
STREAMS = ('left', 'right')
CONNECT = np.ones((3, 3), dtype=bool)


def playable(point):
    """Use the known playing volume. The region behind the learner's base ends here."""
    x, y, z = point
    return bool(abs(x) < 1.30 and -1.24 < y < 1.56 and -.16 < z < 1.6)


def motion_seed(cameras, images):
    """Select a coherent projected trajectory from the temporal foreground."""
    if len(cameras) != len(STREAMS):
        return None
    left, right = (paths(images[key]) for key in STREAMS)
    best = None
    for score_l, px_l in left:
        for score_r, px_r in right:
            count = min(len(px_l), len(px_r))
            paired = {key: px[-count:] for key, px in zip(STREAMS, (px_l, px_r), strict=True)}
            positions = [triangulate_views(cameras, {key: paired[key][i] for key in STREAMS}, max_error=2.5)
                         for i in range(count)]
            if any(point is None or not playable(point) for point in positions):
                continue
            time = np.arange(count, dtype=float)
            design = np.column_stack((np.ones(count), time))
            world = np.array(positions)
            linear = np.linalg.lstsq(design, world, rcond=None)[0]
            error = np.mean(np.linalg.norm(design @ linear - world, axis=1))
            # The learner can exclude its own neutral workspace while it clears.
            if np.max(world[:, 1]) < -.53:
                continue
            confidence = np.sqrt(score_l * score_r) / (1. + error / .015)
            if best is None or confidence > best[0]:
                best = (float(confidence), world, paired)
    if best is not None:
        return best
    return stereo_seed(cameras, images)


def stereo_seed(cameras, images):
    """Pair foreground first when an early change in direction breaks the 2-D path."""
    left, right = (candidates(images[key]) for key in STREAMS)
    a, b = (cameras[key] for key in STREAMS)
    scenes = []
    for px_l, px_r in zip(left, right, strict=True):
        proposals = []
        if len(px_l) and len(px_r):
            rays = b.ray(px_r)
            norm = np.linalg.norm(rays, axis=1)
            for center in px_l:
                plane = np.cross(a.ray(center), b.eye-a.eye)
                scale = np.linalg.norm(plane)
                if scale < 1.e-5:
                    continue
                error = np.abs(rays @ plane) * b.focal / (norm * scale)
                for i in np.flatnonzero(error < 4.8):
                    pixels = dict(left=center, right=px_r[i])
                    point = triangulate_views(cameras, pixels, max_error=2.3)
                    if point is not None and playable(point) and point[1] > -.53:
                        proposals.append((point, pixels, error[i]))
        scenes.append(proposals)
    best = None
    for latest in scenes[-1]:
        for second in scenes[-2]:
            speed = np.linalg.norm(latest[0] - second[0])
            if not .022 < speed < .25:
                continue
            path = [latest, second]
            costs = []
            for scene in reversed(scenes[:-2]):
                target = 2*path[-1][0] - path[-2][0]
                matches = []
                for item in scene:
                    difference = (item[0] - target) / (.038, .065, .19)
                    loss = np.linalg.norm(difference) + .13 * item[2]
                    if loss < 1.45 and np.linalg.norm(item[0] - path[-1][0]) < .25:
                        matches.append((loss, item))
                if not matches:
                    break
                loss, predecessor = min(matches, key=lambda row: row[0])
                path.append(predecessor)
                costs.append(loss)
            if len(path) < 4:
                continue
            samples = np.asarray([item[0] for item in reversed(path)])
            turns = np.diff(samples[:, :2], n=2, axis=0)
            smooth = np.median(np.linalg.norm(turns, axis=1))
            displacement = np.linalg.norm(samples[-1, :2] - samples[0, :2])
            if displacement / (len(path)-1) < .015:
                continue
            confidence = displacement / (.05 + smooth) / (1. + np.mean(costs))
            if best is None or confidence > best[0]:
                pixels = {key: np.array([item[1][key] for item in reversed(path)]) for key in STREAMS}
                best = float(confidence), samples, pixels
    return best


def chromaticity(values):
    rgb = np.asarray(values, dtype=np.float32)
    level = rgb.sum(axis=-1)
    return rgb / np.maximum(1., level[..., None]), level


@dataclass(frozen=True)
class Appearance:
    color: np.ndarray
    tolerance: float
    min_light: float

    @classmethod
    def fit(cls, frames, pixels):
        background = np.median(frames, axis=0)
        color, lights = [], []
        for frame, point in zip(frames[-len(pixels):], pixels, strict=True):
            x, y = np.rint(point).astype(int)
            left, top = max(0, x-4), max(0, y-4)
            box = frame[top:y+5, left:x+5]
            if not box.size:
                continue
            distance = np.max(np.abs(box - background[top:y+5, left:x+5]), axis=-1)
            yy, xx = np.ogrid[top:top+box.shape[0], left:left+box.shape[1]]
            spatial = (yy - point[1])**2 + (xx - point[0])**2
            weight = distance / (1. + .70 * spatial)
            best = np.argmax(weight)
            if weight.flat[best] < 11.:
                continue
            sample = box.reshape(-1, 3)[best]
            color_sample, brightness = chromaticity(sample)
            color.append(color_sample)
            lights.append(float(brightness))
        if len(color) < 3:
            return None
        center = np.median(color, axis=0)
        distance = np.linalg.norm(color - center, axis=1)
        spread = float(np.clip(2.5 * np.median(distance) + .044, .064, .118))
        return cls(center, spread, max(40., .32 * float(np.median(lights))))

    def regions(self, frame, previous, background, expected):
        color, level = chromaticity(frame)
        delta = np.linalg.norm(color - self.color, axis=-1)
        possible = (delta <= self.tolerance) & (level >= self.min_light)
        labels, count = ndimage.label(possible, structure=CONNECT)
        if not count:
            return []
        found = []
        for identity, window in enumerate(ndimage.find_objects(labels, max_label=count), 1):
            slicey, slicex = window
            height, width = slicey.stop-slicey.start, slicex.stop-slicex.start
            if max(height, width) > 21:
                continue
            y, x = np.nonzero(labels[window] == identity)
            if len(x) < 2 or len(x) < .26 * height * width:
                continue
            y, x = y+slicey.start, x+slicex.start
            values = frame[y, x].astype(np.int16)
            changing = np.max(np.abs(values-previous[y, x]), axis=-1)
            foreground = np.max(np.abs(values-background[y, x]), axis=-1)
            if np.max(changing) < 11 and np.median(foreground) < 29:
                continue
            error = delta[y, x]
            weights = np.exp(-.5 * (error / (.68 * self.tolerance))**2)
            center = np.array((x @ weights, y @ weights)) / weights.sum()
            association = 0. if expected is None else min(4., np.linalg.norm(center-expected) / 18.)
            compact = max(0., max(height, width) / max(1., min(height, width)) - 1.8)
            loss = (float(np.median(error)) / self.tolerance + .20 * association
                    + .16 * compact + .26 * np.exp(-np.median(changing) / 28.))
            found.append((loss, center))
        return sorted(found, key=lambda item: item[0])[:7]


class MotionObject:
    """Track only the appearance assigned by the initiating motion sequence."""

    def __init__(self, lenses):
        self.calibration = NaturalViewTransport('fixed', lens_fov=lenses, flight=False)
        self.fits = {}
        self.start = {key: [] for key in STREAMS}
        self.previous = {}
        self.background = {}
        self.appearance = {}
        self.prefix_positions = []
        self.history = deque(maxlen=4)
        self.seed_score = None
        self.position = None

    def _take_seed(self):
        chosen = motion_seed(self.fits, self.start)
        if chosen is None:
            return
        self.seed_score, world, pixels = chosen
        for key in STREAMS:
            self.appearance[key] = Appearance.fit(self.start[key], pixels[key])
            self.background[key] = np.rint(np.median(self.start[key], axis=0)).astype(np.int16)
        if any(value is None for value in self.appearance.values()):
            self.appearance.clear()
            return
        self.prefix_positions = list(zip(range(OPENING-len(world), OPENING), world, strict=True))
        self.history.extend(self.prefix_positions)
        self.position = world[-1]

    def _predict(self, now):
        if not self.history:
            return None, None, 0
        recent, position = self.history[-1]
        gap = now - recent
        if len(self.history) < 2 or gap > 6:
            return None, position, gap
        earlier, previous = self.history[-2]
        step = (position - previous) / (recent-earlier)
        return position + gap * step, position, gap

    def update(self, frames, tick):
        self.fits = self.calibration.observe(frames, tick, self.fits)
        self.position = None
        if tick < OPENING:
            for key in STREAMS:
                self.start[key].append(frames[key])
            if tick == OPENING - 1:
                self._take_seed()
        elif self.appearance and len(self.fits) == 2:
            prediction, last, gap = self._predict(tick)
            choices = {}
            for key in STREAMS:
                expected = None if prediction is None else self.fits[key].project(prediction)[:2]
                choices[key] = self.appearance[key].regions(frames[key], self.previous[key],
                                                            self.background[key], expected)
            best = None
            for loss_l, center_l in choices['left']:
                for loss_r, center_r in choices['right']:
                    centers = {'left': center_l, 'right': center_r}
                    point = triangulate_views(self.fits, centers, max_error=1.8)
                    if point is None or not playable(point):
                        continue
                    if gap <= 5 and np.linalg.norm(point-last) > .155 * gap + .085:
                        continue
                    residual = sum(np.linalg.norm(self.fits[key].project(point)[:2]-centers[key])
                                   for key in STREAMS)
                    motion = 0. if prediction is None else min(3., np.linalg.norm(point-prediction) / (.040+.022*gap))
                    loss = loss_l + loss_r + .30 * residual + .45 * motion
                    if best is None or loss < best[0]:
                        best = (loss, point)
            if best is not None:
                self.position = best[1]
                self.history.append((tick, self.position))
        self.previous = {key: image.astype(np.int16) for key, image in frames.items()}
        return self.position
