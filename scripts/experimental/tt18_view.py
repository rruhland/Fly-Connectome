"""Recover metric view geometry from surveyed, visible landmarks in RGB."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Landmark:
    position: tuple[float, float, float]
    hue: float


# Surveyed hall fixture, in meters from the original tabletop center. The stands
# belong to the hall frame, so moving or resizing the table does not move them.
LANDMARKS = (
    Landmark((-.93, -1.18, .23), .165),
    Landmark((.92, -1.13, .38), .235),
    Landmark((-.96, -.64, .39), .305),
    Landmark((.92, -.68, .15), .374),
    Landmark((-.95, -.12, .17), .435),
    Landmark((.96, -.04, .42), .497),
    Landmark((-.94, .42, .32), .560),
    Landmark((.92, .45, .12), .625),
    Landmark((-.94, .96, .14), .693),
    Landmark((.94, .96, .34), .753),
    Landmark((-.84, 1.46, .43), .818),
    Landmark((.94, 1.48, .24), .977),
)
OBJECT_HUES = {'ball': .073, 'blade': .878}
_CONNECT = np.ones((3, 3), dtype=bool)
_MIN_MARKERS = 7


@dataclass(frozen=True)
class CameraFit:
    """Pinhole camera fitted with a centered principal point and square pixels."""

    eye: np.ndarray
    rotation: np.ndarray
    focal: float
    width: int
    height: int
    rms: float = 0.
    markers: int = 0

    @property
    def center(self):
        return np.array(((self.width - 1) / 2, (self.height - 1) / 2))

    @property
    def basis(self):
        # Match TT4's right, up, forward order.
        return self.rotation * np.array((1., -1., 1.))[:, None]

    def project(self, point):
        camera = np.subtract(point, self.eye) @ self.rotation.T
        with np.errstate(divide='ignore', invalid='ignore'):
            pixel = self.center + self.focal * camera[..., :2] / camera[..., 2:]
        return np.concatenate((pixel, camera[..., 2:]), axis=-1)

    def ray(self, pixel):
        offset = np.subtract(pixel, self.center) / self.focal
        return np.concatenate((offset, np.ones_like(offset[..., :1])), axis=-1) @ self.rotation


@dataclass(frozen=True)
class Blob:
    center: np.ndarray
    pixels: np.ndarray
    weights: np.ndarray


class ImageColors:
    """Small hue-coded features in a native RGB image."""

    def __init__(self, rgb):
        self.shape = rgb.shape[:2]
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError('expected a uint8 RGB image')
        r, g, b = np.moveaxis(rgb.astype(np.float32), -1, 0)
        value = np.maximum(np.maximum(r, g), b)
        self.chroma = value - np.minimum(np.minimum(r, g), b)
        hue = np.where(value == r, g - b, np.where(value == g, b - r, r - g))
        hue /= np.maximum(1., self.chroma)
        hue += np.where(value == r, 0., np.where(value == g, 2., 4.))
        self.hue = np.remainder(hue / 6., 1.)
        self.saturated = self.chroma >= .60 * value
        self._eligible = {}

    def components(self, hue, *, tolerance, minimum_chroma, min_area, max_span, border=False):
        if minimum_chroma not in self._eligible:
            self._eligible[minimum_chroma] = np.nonzero(self.saturated & (self.chroma >= minimum_chroma))
        y, x = self._eligible[minimum_chroma]
        delta = np.abs(self.hue[y, x] - hue)
        match = np.minimum(delta, 1. - delta) <= tolerance
        if match.sum() < min_area:
            return
        x, y = x[match], y[match]
        x0, y0 = x.min(), y.min()
        selected = np.zeros((y.max() - y0 + 1, x.max() - x0 + 1), dtype=bool)
        selected[y - y0, x - x0] = True
        labels, count = ndimage.label(selected, structure=_CONNECT)
        for identity, (ys, xs) in enumerate(ndimage.find_objects(labels, max_label=count), 1):
            if max(ys.stop - ys.start, xs.stop - xs.start) > max_span:
                continue
            if not border and (min(ys.start + y0, xs.start + x0) == 0 or
                               ys.stop + y0 == self.shape[0] or xs.stop + x0 == self.shape[1]):
                continue
            y, x = np.nonzero(labels[ys, xs] == identity)
            if len(x) < min_area:
                continue
            x, y = x + xs.start + x0, y + ys.start + y0
            weights = self.chroma[y, x]
            points = np.column_stack((x, y))
            yield Blob(np.average(points, axis=0, weights=weights), points, weights)


def _landmark_pixels(colors, landmarks):
    for landmark in landmarks:
        regions = tuple(colors.components(landmark.hue, tolerance=.025, minimum_chroma=98.,
                                          min_area=3, max_span=max(18, colors.shape[0] // 8)))
        if len(regions) == 1:
            yield landmark.position, regions[0].center


def _matrix_seed(points, pixels, shape):
    """Normalized direct linear transform used only to seed the pinhole fit."""
    height, width = shape
    center = (np.array((width, height)) - 1) / 2
    origin = points.mean(axis=0)
    scale = np.linalg.norm(points - origin, axis=1).mean()
    if scale < 1e-7:
        return None
    world = np.column_stack(((points - origin) / scale, np.ones(len(points))))
    u, v = ((pixels - center) / height).T
    zero = np.zeros_like(world)
    design = np.concatenate((np.column_stack((world, zero, -u[:, None] * world)),
                             np.column_stack((zero, world, -v[:, None] * world))))
    _, singular, vectors = np.linalg.svd(design, full_matrices=False)
    if singular[-2] < .001 * singular[0]:
        return None
    matrix = vectors[-1].reshape(3, 4)
    matrix *= np.sign(np.linalg.det(matrix[:, :3])) / np.linalg.norm(matrix[2, :3])
    focal = np.linalg.norm(matrix[:2, :3], axis=1).mean()
    if not .18 < focal < 6.:
        return None
    matrix[:2] /= focal
    left, _, right = np.linalg.svd(matrix[:, :3])
    rotation = left @ right
    eye = origin - scale * (rotation.T @ matrix[:, 3])
    return np.r_[Rotation.from_matrix(rotation).as_rotvec(), eye, np.log(focal * height)]


def _fit_camera(points, pixels, shape, previous):
    height, width = shape

    def camera(parameters):
        return CameraFit(parameters[3:6], Rotation.from_rotvec(parameters[:3]).as_matrix(),
                         float(np.exp(parameters[6])), width, height)

    def residual(parameters):
        return (camera(parameters).project(points)[:, :2] - pixels).ravel()

    initial = _matrix_seed(points, pixels, shape)
    if initial is None:
        return None
    starts = [initial]
    if previous is not None:
        starts.append(np.r_[Rotation.from_matrix(previous.rotation).as_rotvec(),
                            previous.eye, np.log(previous.focal)])
    best = None
    bounds = (np.r_[np.full(6, -np.inf), np.log(height * .18)],
              np.r_[np.full(6, np.inf), np.log(height * 6.)])
    for start in starts:
        if np.any(camera(start).project(points)[:, 2] <= .02):
            continue
        fit = least_squares(residual, start, bounds=bounds, loss='soft_l1', f_scale=.40, max_nfev=55)
        view = camera(fit.x)
        image = view.project(points)
        errors = np.linalg.norm(image[:, :2] - pixels, axis=1)
        rms = float(np.sqrt(np.mean(errors**2)))
        if np.any(image[:, 2] <= .02) or rms > .90 or np.max(errors) > 1.8:
            continue
        if best is None or rms < best.rms:
            best = CameraFit(view.eye, view.rotation, view.focal, width, height, rms, len(points))
    return best


class ViewRegistry:
    """Track RGB stream registrations and discard unverifiable current views."""

    def __init__(self, landmarks=LANDMARKS, *, motion_threshold=.56):
        self.landmarks = tuple(landmarks)
        self.motion_threshold = motion_threshold
        self.known = {}
        self.active = {}
        self.counts = {'frames': 0, 'fit_attempts': 0, 'fits': 0, 'unregistered_views': 0,
                       'added': 0, 'removed': 0, 'relocated': 0}
        self.last_events = {}
        self._streams = set()

    def update(self, frames):
        """Consume synchronized RGB keyed by opaque, stable stream identifiers."""
        return self.observe({key: ImageColors(rgb) for key, rgb in frames.items()})

    def observe(self, frames):
        self.counts['frames'] += 1
        streams = set(frames)
        self.last_events = {key: 'removed' for key in self._streams - streams}
        self.counts['removed'] += len(self._streams - streams)
        self.counts['added'] += len(streams - self._streams)
        self._streams = streams
        active = {}
        for key, colors in frames.items():
            prior = self.known.get(key)
            observations = tuple(_landmark_pixels(colors, self.landmarks))
            if len(observations) < _MIN_MARKERS:
                self.last_events[key] = 'insufficient_landmarks'
                continue
            position, pixel = (np.asarray(values) for values in zip(*observations, strict=True))
            if prior is not None and (prior.height, prior.width) == colors.shape:
                error = np.linalg.norm(prior.project(position)[:, :2] - pixel, axis=1)
                if np.sqrt(np.mean(error**2)) <= self.motion_threshold and max(error) <= 1.4:
                    active[key] = prior
                    self.last_events[key] = 'verified'
                    continue
            self.counts['fit_attempts'] += 1
            fit = _fit_camera(position, pixel, colors.shape, prior)
            if fit is None:
                self.last_events[key] = 'rejected'
                continue
            self.counts['fits'] += 1
            self.counts['relocated'] += prior is not None
            self.last_events[key] = 'added' if prior is None else 'relocated'
            active[key] = self.known[key] = fit
        self.counts['unregistered_views'] += len(frames) - len(active)
        self.active = active
        return active


def object_blobs(frames):
    """Detect two explicitly color-coded roles in each RGB view."""
    result = {name: {} for name in OBJECT_HUES}
    for key, frame in frames.items():
        colors = frame if isinstance(frame, ImageColors) else ImageColors(frame)
        for name, hue in OBJECT_HUES.items():
            ball = name == 'ball'
            blobs = tuple(colors.components(hue, tolerance=.049 if ball else .042,
                                             minimum_chroma=51. if ball else 62.,
                                             min_area=1 if ball else 4,
                                             max_span=max(10, colors.shape[0] // (10 if ball else 2))))
            if ball:
                # Wood edges and reflections may have the same hue at far lower chroma.
                brightest = max(blobs, key=lambda blob: blob.weights.max(), default=None)
                if brightest is not None and brightest.weights.max() >= 91.:
                    result[name][key] = brightest
            elif blobs:
                result[name][key] = max(blobs, key=lambda blob: blob.weights.sum())
    return result


def object_pixels(frames):
    return {name: {key: blob.center for key, blob in views.items()}
            for name, views in object_blobs(frames).items()}


def _linear_rays(cameras, pixels):
    directions = np.array([view.ray(pixel) for view, pixel in zip(cameras, pixels, strict=True)])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    origins = np.array([view.eye for view in cameras])
    perpendicular = np.eye(3) - directions[:, :, None] * directions[:, None, :]
    system = perpendicular.sum(axis=0)
    if np.linalg.eigvalsh(system)[0] < .001:
        return None
    return np.linalg.solve(system, (perpendicular @ origins[:, :, None]).sum(axis=0).ravel())


def triangulate_views(fits, pixels, *, max_error=2.):
    """Intersect synchronized rays and abstain on depth without enough agreement."""
    views = tuple(key for key in fits if key in pixels)
    if len(views) < 2:
        return None
    camera = tuple(fits[key] for key in views)
    pixel = np.array([pixels[key] for key in views])
    best = None
    for first, second in combinations(range(len(views)), 2):
        indices = (first, second)
        point = _linear_rays([camera[index] for index in indices], pixel[list(indices)])
        if point is None:
            continue
        projection = np.array([view.project(point) for view in camera])
        error = np.linalg.norm(projection[:, :2] - pixel, axis=1)
        inliers = (projection[:, 2] > .02) & (error <= max_error)
        if inliers.sum() < max(2, len(views) - 1):
            continue
        score = int(inliers.sum()), -float(np.mean(error[inliers]))
        if best is None or score > best[0]:
            best = score, inliers, point
    if best is None:
        return None
    _, inliers, point = best
    if inliers.sum() > 2:
        point = _linear_rays([view for view, valid in zip(camera, inliers, strict=True) if valid], pixel[inliers])
    return point
