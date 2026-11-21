"""Calibrated player-eye and spectator RGB cameras for both table simulators."""

from collections import deque
from dataclasses import dataclass
from functools import cached_property, lru_cache
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .tt3_arena import (BALL_RADIUS, HZ, IMAGE_WIDTH, NET_HEIGHT, PADDLE_RADIUS,
                        PIXELS_PER_METER, TABLE_HALF_LENGTH, TABLE_HALF_WIDTH,
                        TT3Arena)
from .tt3_visual import Track


@dataclass(frozen=True)
class PerspectiveCamera:
    """Pinhole calibration in meters, with x lateral, y down-court and z up."""

    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    width: int
    height: int
    fovy: float

    @cached_property
    def basis(self):
        forward = _unit(np.subtract(self.target, self.eye))
        right = _unit(np.cross(forward, (0., 0., 1.)))
        return np.array((right, np.cross(right, forward), forward))

    @cached_property
    def focal(self):
        return self.height / (2 * math.tan(math.radians(self.fovy) / 2))

    @cached_property
    def center(self):
        return (self.width - 1) / 2, (self.height - 1) / 2

    def project(self, point):
        """Return horizontal and vertical pixel coordinates and optical depth."""
        x, y, depth = np.subtract(point, self.eye) @ self.basis.T
        with np.errstate(divide='ignore', invalid='ignore'):
            return (self.center[0] + self.focal * x / depth,
                    self.center[1] - self.focal * y / depth, depth)

    def ray(self, pixel):
        """Return a world direction with one unit of optical depth per step."""
        coordinates = np.subtract(pixel, self.center) / self.focal
        x, y = np.moveaxis(coordinates, -1, 0)
        return np.stack((x, -y, np.ones_like(x)), axis=-1) @ self.basis

    @cached_property
    def rays(self):
        y, x = np.indices((self.height, self.width))
        center_x, center_y = self.center
        return np.stack(((x - center_x) / self.focal, (center_y - y) / self.focal,
                         np.ones_like(x)), axis=-1) @ self.basis


def _unit(vector):
    return vector / np.linalg.norm(vector)


def cameras(view='first_person', width=96, height=72, *, baseline=.16):
    """Return a parallel eye pair, a single central eye, or an overhead/side pair."""
    if type(width) is not int or type(height) is not int or min(width, height) < 16:
        raise ValueError('camera dimensions must be integers of at least 16 pixels')
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError('baseline must be positive')
    if view == 'first_person' or view == 'mono':
        offsets = (-baseline / 2, baseline / 2) if view == 'first_person' else (0.,)
        return tuple(PerspectiveCamera((x, -2.30, .84), (x, .32, .16), width, height, 65.)
                     for x in offsets)
    if view == 'overhead_side':
        return (PerspectiveCamera((0., -.12, 3.4), (0., 0., 0.), width, height, 55.),
                PerspectiveCamera((3.35, -.36, .75), (0., 0., .27), width, height, 52.))
    raise ValueError(f'unknown camera view {view!r}')


def _intersection(rig, left, right):
    first, second = rig
    a, b = _unit(first.ray(left)), _unit(second.ray(right))
    delta = np.subtract(second.eye, first.eye)
    cross = np.cross(a, b)
    angle = float(cross @ cross)
    if angle < 1e-9:
        return None
    t = float(np.cross(delta, b) @ cross) / angle
    s = float(np.cross(delta, a) @ cross) / angle
    if min(t, s) <= 0.:
        return None
    p, q = first.eye + t * a, second.eye + s * b
    return (p + q) / 2, float(np.linalg.norm(p - q))


def triangulate(rig, left, right):
    """Intersect two sight lines by taking the midpoint of their closest approach."""
    result = _intersection(rig, left, right)
    return None if result is None else tuple(result[0])


def _divide(numerator, denominator):
    with np.errstate(divide='ignore', invalid='ignore'):
        return numerator / denominator


def _paint(image, depth, distances, selected, rgb):
    mask = selected & (distances > .05) & (distances < depth)
    depth[mask] = distances[mask]
    image[mask] = rgb if np.ndim(rgb) == 1 else rgb[mask]


def _box(camera, image, depth, low, high, rgb):
    eye = np.asarray(camera.eye)
    near = _divide(np.subtract(low, eye), camera.rays)
    far = _divide(np.subtract(high, eye), camera.rays)
    start = np.min((near, far), axis=0).max(axis=-1)
    end = np.max((near, far), axis=0).min(axis=-1)
    face = np.argmax(np.minimum(near, far), axis=-1)
    shade = np.array((.76, .90, 1.))[face][..., None]
    _paint(image, depth, start, start < end, np.asarray(rgb) * shade)


@lru_cache(maxsize=24)
def background(camera):
    """Rasterize the empty room and regulation table once for a fixed camera."""
    rays, eye = camera.rays, np.asarray(camera.eye)
    shape = (camera.height, camera.width)
    image = np.empty((*shape, 3), dtype=np.uint8)
    image[...] = (163, 168, 174)
    depth = np.full(shape, np.inf)
    floor = -.76
    for axis, extent in ((0, 3.7), (1, 4.6)):
        for sign in (-1, 1):
            distances = _divide(sign * extent - eye[axis], rays[..., axis])
            heights = eye[2] + distances * rays[..., 2]
            panel = np.clip((heights - floor) / .88, 0., 1.)[..., None]
            rgb = np.array((104, 111, 119)) + panel * np.array((60, 59, 59))
            _paint(image, depth, distances, heights > floor, rgb)
    distances = _divide(floor - eye[2], rays[..., 2])
    x, y = (eye[axis] + distances * rays[..., axis] for axis in range(2))
    with np.errstate(invalid='ignore'):
        seam = (np.abs(x - np.rint(x)) < .008) | (np.abs(y - np.rint(y)) < .008)
    shade = (14 * np.exp(-.08 * (x**2 + y**2)) - 6 * seam)[..., None]
    _paint(image, depth, distances, np.ones(shape, dtype=bool), np.array((105, 106, 103)) + shade)

    a, b = TABLE_HALF_WIDTH, TABLE_HALF_LENGTH
    for x_pos in (-.58, .58):
        for y_pos in (-1.02, 1.02):
            _box(camera, image, depth, (x_pos - .027, y_pos - .027, floor),
                 (x_pos + .027, y_pos + .027, -.028), (66, 72, 78))
    _box(camera, image, depth, (-a, -b, -.049), (a, b, -.0002), (45, 55, 61))
    distances = _divide(-eye[2], rays[..., 2])
    x, y = (eye[axis] + distances * rays[..., axis] for axis in range(2))
    inside = (np.abs(x) <= a) & (np.abs(y) <= b)
    stripes = inside & ((np.abs(x) >= a - .020) | (np.abs(y) >= b - .020) | (np.abs(x) < .003))
    color = np.full((*shape, 3), (51, 68, 75), dtype=np.uint8)
    color[stripes] = (208, 211, 212)
    _paint(image, depth, distances, inside, color)

    # The net has openings, so a passing object can remain visible through it.
    net_half_width = TABLE_HALF_WIDTH + .1525
    for x_pos in (-net_half_width, net_half_width):
        _box(camera, image, depth, (x_pos - .009, -.010, -.026),
             (x_pos + .009, .010, NET_HEIGHT + .006), (45, 48, 55))
    distances = _divide(-eye[1], rays[..., 1])
    x, z = (eye[axis] + distances * rays[..., axis] for axis in (0, 2))
    inside = (np.abs(x) <= net_half_width) & (z >= 0.) & (z <= NET_HEIGHT)
    with np.errstate(invalid='ignore'):
        mesh = ((np.remainder(x, .027) < .007) | (np.remainder(z, .027) < .007)) & inside
    _paint(image, depth, distances, mesh, (67, 72, 79))
    _paint(image, depth, distances, inside & (z >= NET_HEIGHT - .012), (221, 222, 218))
    image.flags.writeable = depth.flags.writeable = False
    return image, depth


def _patch(camera, points, margin):
    left = max(0, math.floor(min(point[0] for point in points) - margin))
    right = min(camera.width, math.ceil(max(point[0] for point in points) + margin) + 1)
    top = max(0, math.floor(min(point[1] for point in points) - margin))
    bottom = min(camera.height, math.ceil(max(point[1] for point in points) + margin) + 1)
    return slice(top, max(top, bottom)), slice(left, max(left, right))


def _segment(camera, image, depth, start, end, radius, color):
    a, b = camera.project(start), camera.project(end)
    if min(a[2], b[2]) <= .05:
        return
    area = _patch(camera, (a, b), camera.focal * radius / min(a[2], b[2]))
    y, x = np.ogrid[area]
    dx, dy = b[0] - a[0], b[1] - a[1]
    fraction = np.clip(((x - a[0]) * dx + (y - a[1]) * dy) / max(dx * dx + dy * dy, 1e-6), 0, 1)
    distances = 1 / ((1 - fraction) / a[2] + fraction / b[2])
    radial = (x - a[0] - fraction * dx)**2 + (y - a[1] - fraction * dy)**2
    _paint(image[area], depth[area], distances - radius,
           radial <= (camera.focal * radius / distances)**2, color)


def _paddle(camera, image, depth, point, normal, radius, color):
    center = camera.project(point)
    if center[2] <= radius + .05:
        return
    area = _patch(camera, (center,), camera.focal * radius / (center[2] - radius) + 1)
    rays = camera.rays[area]
    normal = np.asarray(normal)
    distances = _divide(np.subtract(point, camera.eye) @ normal, rays @ normal)
    with np.errstate(invalid='ignore'):
        offset = rays * distances[..., None] - np.subtract(point, camera.eye)
        squared = np.sum(offset * offset, axis=-1)
    _paint(image[area], depth[area], distances, squared <= (radius - .004)**2, color)
    _paint(image[area], depth[area], distances, squared <= radius**2, (44, 45, 47))


def _ball(camera, image, depth, point, radius, color):
    x, y, distance = camera.project(point)
    if distance <= radius + .05:
        return
    relative = camera.focal * radius / distance
    area = _patch(camera, ((x, y),), relative + 1.)
    yy, xx = np.ogrid[area]
    dx, dy = np.abs(xx - x), np.abs(yy - y)
    if relative <= .85:
        # A compact optical point-spread function preserves the subpixel center.
        coverage = np.minimum(1., math.pi * relative**2 * np.maximum(0, 1 - dx) * np.maximum(0, 1 - dy))
    else:
        coverage = np.clip(relative + .5 - np.hypot(dx, dy), 0., 1.)
    alpha = np.where(distance - radius < depth[area], coverage, 0.)[..., None]
    patch = image[area]
    patch[...] = np.rint(patch + alpha * (np.asarray(color) - patch))


def render_scene(rig, arm, opponent, ball, colors, *, decoy=None, ball_radius=.02, paddle_radius=.083):
    """Render the court and moving geometry as uint8 RGB through the supplied lenses."""
    result = []
    for camera in rig:
        backdrop, static_depth = background(camera)
        image, depth = backdrop.copy(), static_depth.copy()
        for pose, color in zip((arm, opponent), colors):
            shoulder, elbow, blade, normal = pose
            _segment(camera, image, depth, (shoulder[0], shoulder[1], -.76), shoulder,
                     .080, (81, 84, 89))
            _segment(camera, image, depth, shoulder, elbow, .044, (118, 120, 123))
            handle = np.subtract(elbow, blade)
            along_face = handle - (handle @ normal) * np.asarray(normal)
            attachment = blade + along_face * min(1., paddle_radius / max(np.linalg.norm(along_face), 1e-6))
            _segment(camera, image, depth, elbow, attachment, .027, (140, 142, 145))
            _paddle(camera, image, depth, blade, normal, paddle_radius, color)
        if decoy is not None:
            _ball(camera, image, depth, decoy, ball_radius, colors[3])
        _ball(camera, image, depth, ball, ball_radius, colors[2])
        result.append(image)
    return tuple(result)


class TT3PerspectiveArena(TT3Arena):
    """Original TT3 mechanics observed through the same calibrated cameras as TT4."""

    def __init__(self, seed=0, *, view='first_person', width=96, height=72, baseline=.16, **kwargs):
        self.cameras = cameras(view, width, height, baseline=baseline)
        super().__init__(seed, **kwargs)

    def frame(self):
        decoy = None
        if self.distractor:
            t = self.frames / HZ
            px, py, pz = self._decoy_phases
            decoy = (.61 * math.sin(1.4 * t + px), 1.42 * math.sin(3.1 * t + py),
                     .45 + .31 * math.sin(4.7 * t + pz))
        x, y, z = self.rival
        rival_arm = ((0., 1.86, .11), (.55 * x, (y + 1.86) / 2, z + .08), self.rival, self.rival_normal)
        return render_scene(self.cameras, self._pose, rival_arm, self.ball, self._colors,
                            decoy=decoy, ball_radius=BALL_RADIUS, paddle_radius=PADDLE_RADIUS)


@dataclass(frozen=True)
class ImageDetection:
    position: tuple[float, float]
    appearance: tuple[float, ...]
    area: float


def _color_regions(chroma, hue, selected):
    y, x = np.nonzero(selected)
    indices = np.full(selected.shape, -1, dtype=int)
    indices[y, x] = np.arange(len(x))
    directions = hue[y, x]
    sources, targets = [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        yy, xx = y + dy, x + dx
        within = (yy < selected.shape[0]) & (xx >= 0) & (xx < selected.shape[1])
        source = np.flatnonzero(within)
        target = indices[yy[within], xx[within]]
        present = target >= 0
        source, target = source[present], target[present]
        similar = np.sum(directions[source] * directions[target], axis=1) > .79
        sources.append(source[similar])
        targets.append(target[similar])
    source, target = np.concatenate(sources), np.concatenate(targets)
    graph = coo_matrix((np.ones(len(source)), (source, target)), shape=(len(x), len(x)))
    groups, labels = connected_components(graph, directed=False)
    weights = chroma[y, x].astype(float)
    total = np.bincount(labels, weights=weights, minlength=groups)
    px = np.bincount(labels, weights=x * weights, minlength=groups) / total
    py = np.bincount(labels, weights=y * weights, minlength=groups) / total
    peak = np.zeros(groups)
    np.maximum.at(peak, labels, weights)
    areas = np.bincount(labels, minlength=groups)
    hues = np.stack([np.bincount(labels, weights=channel * weights, minlength=groups)
                     for channel in directions.T], axis=-1)
    return [ImageDetection((px[i], py[i]), tuple(_unit(hues[i])), float(areas[i]))
            for i in np.flatnonzero((peak >= 16) & (areas <= .20 * selected.size))]


def _color_detector(camera):
    # This is a fixed RGB background calibration. It contains the table and room only.
    reference = background(camera)[0].astype(np.int16)

    def detect(frame):
        frame = np.asarray(frame)
        if frame.shape != reference.shape or frame.dtype != np.uint8:
            raise ValueError('expected a calibrated uint8 RGB camera image')
        difference = frame.astype(np.int16) - reference
        chroma = np.ptp(difference, axis=-1)
        colored = (chroma >= 7) & (np.ptp(frame, axis=-1) >= 16)
        if not colored.any():
            return []
        hue = difference - difference.mean(axis=-1, keepdims=True)
        hue /= np.maximum(1., np.linalg.norm(hue, axis=-1, keepdims=True))
        return _color_regions(chroma, hue, colored)

    return detect


class _Trajectory:
    def __init__(self, index, tick, position, appearance, area):
        self.id = index
        self.history = deque(maxlen=5)
        self.age = 0
        self.appearance = appearance
        self.area = area
        self.observe(tick, position, appearance, area)

    def observe(self, tick, position, appearance, area):
        self.history.append((tick, position))
        self.tick, self.position = self.history[-1]
        first_tick, first_position = self.history[0]
        self.velocity = (self.position - first_position) / max(1, tick - first_tick)
        self.appearance = .75 * self.appearance + .25 * appearance
        self.current_area = area
        self.area = .8 * self.area + .2 * area
        self.age += 1

    def predict(self, tick):
        return self.position + (tick - self.tick) * self.velocity

    @property
    def color(self):
        red, green, blue = self.appearance[:3]
        angle = math.atan2(math.sqrt(3.) * (green - blue), 2 * red - green - blue)
        return round(12 * angle / math.tau) % 12


class PerspectiveTracker:
    """Associate anonymous image observations and triangulate calibrated 3D tracks."""

    def __init__(self, rig, *, coordinates='tt3', detector_factory=None, pair_area_weight=0.,
                 max_epipolar_error=None):
        if len(rig) != 2 or coordinates not in ('tt3', 'world'):
            raise ValueError('tracking requires two calibrated eyes and a known coordinate convention')
        self.rig = tuple(rig)
        self.coordinates = coordinates
        self.pair_area_weight = pair_area_weight
        self.max_epipolar_error = max_epipolar_error
        self._detector_factory = _color_detector if detector_factory is None else detector_factory
        self.reset()

    def reset(self):
        self.detectors = tuple(self._detector_factory(camera) for camera in self.rig)
        self.tick = -1
        self.count = 0
        self._active = []
        self.support = {}

    @property
    def hypotheses(self):
        return {track.id: track for track in self._active}

    @property
    def appearances(self):
        return {track.id: track.color for track in self._active}

    @property
    def _top(self):
        return self

    def _matches(self, frames):
        left, right = (tuple(detector(frame)) for detector, frame in zip(self.detectors, frames))
        if not left or not right:
            return []
        first, second = self.rig
        a = first.ray([item.position for item in left])[:, None, :]
        b = second.ray([item.position for item in right])[None, :, :]
        cross = np.cross(a, b)
        base = np.subtract(second.eye, first.eye)
        angle = np.sum(cross * cross, axis=-1)
        valid_angle = angle >= 1e-9
        denominator = np.maximum(angle, 1e-9)
        t = np.sum(np.cross(base, b) * cross, axis=-1) / denominator
        s = np.sum(np.cross(base, a) * cross, axis=-1) / denominator
        points_l, points_r = first.eye + t[..., None] * a, second.eye + s[..., None] * b
        positions = (points_l + points_r) / 2
        gaps = np.linalg.norm(points_l - points_r, axis=-1)
        features_l = np.array([item.appearance for item in left])
        features_r = np.array([item.appearance for item in right])
        mismatch = np.linalg.norm(features_l[:, None, :] - features_r[None, :, :], axis=-1)
        valid = valid_angle & (t > 0.) & (s > 0.) & (mismatch <= .65) & (gaps <= .095)
        valid &= np.all(np.abs(positions) <= (1.75, 2.65, 2.05), axis=-1)
        if self.max_epipolar_error is not None:
            pixels = gaps * (first.focal + second.focal) / np.maximum(t + s, .001)
            valid &= pixels <= self.max_epipolar_error
        costs = mismatch + 5. * gaps
        if self.pair_area_weight:
            area_l = np.log2([max(item.area, .001) for item in left])
            area_r = np.log2([max(item.area, .001) for item in right])
            costs += self.pair_area_weight * np.abs(area_l[:, None] - area_r[None, :])
        costs = np.where(valid, costs, 100.)
        rows, columns = linear_sum_assignment(costs)
        return [(positions[row, column], (features_l[row] + features_r[column]) / 2, left[row].area)
                for row, column in zip(rows, columns) if valid[row, column]]

    def _coordinates(self, vector):
        if self.coordinates == 'world':
            return tuple(vector)
        x, y, z = vector
        return tuple(np.array((y, x, -z)) * (PIXELS_PER_METER / IMAGE_WIDTH))

    def update(self, frames):
        if len(frames) != 2:
            raise ValueError('expected two RGB frames in calibration order')
        self.tick += 1
        sightings = self._matches(frames)
        used = set()
        if self._active and sightings:
            positions, appearances, _ = zip(*sightings)
            predictions = np.array([track.predict(self.tick) for track in self._active])
            distance = np.linalg.norm(predictions[:, None, :] - positions, axis=-1)
            hues = np.array([track.appearance for track in self._active])
            mismatch = np.linalg.norm(hues[:, None, :] - appearances, axis=-1)
            costs = distance + .5 * mismatch
            costs[(distance > .55) | (mismatch > .70)] = 100.
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if costs[row, column] < 100.:
                    self._active[row].observe(self.tick, *sightings[column])
                    used.add(column)
        for index, sighting in enumerate(sightings):
            if index not in used:
                self._active.append(_Trajectory(self.count, self.tick, *sighting))
                self.count += 1
        self._active = [track for track in self._active if self.tick - track.tick <= 5]
        self.support = {track.id: (track.current_area, track.area) for track in self._active}
        return tuple(Track(track.id, self._coordinates(track.predict(self.tick)), self._coordinates(track.velocity),
                           track.tick == self.tick, track.age) for track in self._active)
