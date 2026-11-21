"""Track the ball and moving racket from the regulation hall's native RGB views."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .tt3_visual import Track
from .tt4_arena import BALL_RADIUS, PADDLE_RADIUS
from .tt4_native_temporal import modal_image


_CONNECT = np.ones((3, 3))


@dataclass(frozen=True)
class _Blob:
    center: np.ndarray
    size: np.ndarray
    area: float
    appearance: np.ndarray
    change: float


class _Stereo:
    def __init__(self, rig):
        self.camera = rig[0]
        self.baseline = np.linalg.norm(np.subtract(rig[1].eye, rig[0].eye))
        if (len(rig) != 2 or rig[0].width != rig[1].width or rig[0].height != rig[1].height
                or not np.allclose(rig[0].basis, rig[1].basis)):
            raise ValueError('hall tracking requires calibrated parallel stereo cameras')
        direction = np.subtract(rig[1].eye, rig[0].eye) / self.baseline
        if not np.allclose(direction, self.camera.basis[0]):
            raise ValueError('hall tracking requires horizontally separated eyes')
        self.rig = tuple(rig)
        self.focal = self.camera.focal

    def position(self, left, right):
        disparity = left[0] - right[0]
        if disparity < 2:
            return None
        depth = self.baseline * self.focal / disparity
        pixel = left[0], (left[1] + right[1]) * .5
        return np.asarray(self.camera.eye) + depth * self.camera.ray(pixel)

    def pixels(self, position):
        return np.array([camera.project(position)[:2] for camera in self.rig])


def _contrast(current, reference):
    delta = np.abs(current - reference)
    return np.maximum(np.maximum(delta[..., 0], delta[..., 1]), delta[..., 2])


def _blobs(rgb, difference, recent):
    labels, count = ndimage.label(difference >= 11, structure=_CONNECT)
    regions = ndimage.find_objects(labels, max_label=count)
    blobs = []
    for identity, (ys, xs) in enumerate(regions, 1):
        size = np.array((xs.stop - xs.start, ys.stop - ys.start))
        if max(size) > 15 or min(size) / max(size) < .25:
            continue
        y, x = np.nonzero(labels[ys, xs] == identity)
        y += ys.start
        x += xs.start
        contrast = difference[y, x]
        if max(contrast) < 15 or len(x) < .33 * size.prod():
            continue
        weight = contrast.astype(float)
        center = np.array((x @ weight, y @ weight)) / weight.sum()
        color = rgb[y[np.argmax(contrast)], x[np.argmax(contrast)]]
        color = color / max(20., np.linalg.norm(color))
        change = .65 if recent is None else min(1., np.max(recent[y, x]) / 45.)
        blobs.append(_Blob(center, size, float(len(x)), color, change))
    return blobs


def _response_apertures(histories, images):
    designs, measurements = [], []
    offset = 0
    for history in histories:
        requested = np.cumsum([action for action, _ in history], axis=0)
        effective = np.zeros_like(requested, dtype=float)
        for tick in range(1, len(effective)):
            effective[tick] = .55 * effective[tick - 1] + .45 * requested[tick]
        effective = effective[::2]
        count = len(effective)
        if count >= 4:
            time = np.linspace(-1., 1., count)
            drift, _ = np.linalg.qr(np.column_stack((np.ones(count), time, time**2)))
            designs.append(effective - drift @ (drift.T @ effective))
            measurements.append(images[offset:offset + count])
        offset += count
    if not designs:
        raise ValueError('hall tracking needs a probe tape with varying actions')
    design = np.concatenate(designs).astype(np.float32)
    if not np.any(design):
        raise ValueError('hall tracking needs action-responsive probe images')
    rgb = np.concatenate(measurements).astype(np.float32)
    shape = rgb.shape[1:]
    response = (design.T @ rgb.reshape(len(rgb), -1)).reshape((design.shape[1], *shape))
    response = np.sqrt(np.square(response).sum(axis=(0, 4)))
    apertures = []
    for eye in response:
        local = ndimage.gaussian_filter(eye, 3.)
        standout = local > .30 * local.max()
        labels, count = ndimage.label(standout, structure=_CONNECT)
        regions = ndimage.find_objects(labels, max_label=count)
        if not regions:
            raise ValueError('probe images have no repeatable action response')
        y, x = np.unravel_index(np.argmax(local), local.shape)
        ys, xs = regions[labels[y, x] - 1]
        height, width = eye.shape
        apertures.append((slice(max(0, ys.start - 18), min(height, ys.stop + 18)),
                          slice(max(0, xs.start - 18), min(width, xs.stop + 18))))
    return tuple(apertures)


def _material(image, difference, aperture):
    crop = image[aperture]
    foreground = difference[aperture] >= 28
    broad = ndimage.binary_opening(foreground, structure=np.ones((1, 6)))
    pixels = crop[broad]
    if len(pixels) < 16:
        pixels = crop[foreground]
    if len(pixels) < 16:
        return None
    bins = pixels.astype(np.int32) // 16
    ids = bins @ np.array((256, 16, 1))
    mode = np.bincount(ids).argmax()
    return np.median(pixels[ids == mode], axis=0)


@dataclass(frozen=True)
class _Face:
    center: np.ndarray
    area: int
    top: int
    middle: np.ndarray
    span: np.ndarray
    profile: np.ndarray


def _face(image, reference, palette, prior, aperture, *, motion_supported=False):
    height, width = image.shape[:2]
    if prior is None:
        ys, xs = aperture
        x0, x1, y0, y1 = xs.start, xs.stop, ys.start, ys.stop
    else:
        center, radius = prior
        reach = max(16, 2.65 * radius)
        x0, y0 = np.maximum(0, np.floor(center - reach)).astype(int)
        x1, y1 = np.minimum((width, height), np.ceil(center + reach)).astype(int)
    crop = image[y0:y1, x0:x1].astype(np.float32)
    if not crop.size:
        return None
    length = np.linalg.norm(palette)
    projection = crop @ (palette / length)
    chroma = np.maximum(0., np.einsum('ijk,ijk->ij', crop, crop) - projection**2)
    # Racket rubber has a consistent hue through shading. Its intensity still
    # limits which dark, similarly colored edges of the hall can join it.
    matches = ((chroma <= (.105 * projection + 2)**2)
               & (projection > .24 * length) & (projection < 1.67 * length))
    if motion_supported:
        # Trim same-hue hall furniture while preserving the partially covered edge.
        novelty = _contrast(crop, reference[y0:y1, x0:x1])
        matches &= ndimage.binary_dilation(novelty >= 15, iterations=2)
    labels, count = ndimage.label(matches, structure=_CONNECT)
    best = None
    for identity, (ys, xs) in enumerate(ndimage.find_objects(labels, max_label=count), 1):
        size = xs.stop - xs.start, ys.stop - ys.start
        if size[0] < 6 or not 2 <= size[1] <= size[0] * 1.55:
            continue
        rows, columns = np.nonzero(labels[ys, xs] == identity)
        if len(columns) < size[0] * size[1] * .40:
            continue
        top = y0 + ys.start
        center = np.array((x0 + xs.start + (size[0] - 1) / 2, top + (size[1] - 1) / 2))
        score = len(columns)
        if prior is not None:
            pixel, radius = prior
            score /= 1. + (np.linalg.norm(center - pixel) / max(7., radius)) ** 2
        if best is not None and score <= best[0]:
            continue
        low = np.full(size[1], size[0], dtype=float)
        high = np.full(size[1], -1., dtype=float)
        np.minimum.at(low, rows, columns)
        np.maximum.at(high, rows, columns)
        middle = x0 + xs.start + .5 * (low + high)
        span = high - low + 1
        best = score, center, len(columns), top, middle, span, identity, ys, xs
    if best is None:
        return None
    _, center, area, top, middle, span, identity, ys, xs = best
    mask = labels == identity
    profile = _vertical_profile(crop, reference[y0:y1, x0:x1], mask, center - (x0, y0), xs, ys)
    profile[:, 0] += x0
    return _Face(center, area, top, middle, span, profile)


def _vertical_profile(crop, reference, mask, center, xs, ys):
    x0, x1 = max(0, xs.start - 1), min(crop.shape[1], xs.stop + 1)
    y0, y1 = max(0, ys.start - 2), min(crop.shape[0], ys.stop + 2)
    region = np.s_[y0:y1, x0:x1]
    source = crop[region]
    background = reference[region]
    material = np.median(source[mask[region]], axis=0)
    foreground = source - background
    expected = material - background
    divisor = np.einsum('ijk,ijk->ij', expected, expected)
    alpha = np.einsum('ijk,ijk->ij', foreground, expected) / np.maximum(divisor, 1)
    # A tight neighborhood avoids other moving pixels along the arm.
    alpha *= ndimage.binary_dilation(mask[region], iterations=2)
    valid = (alpha > .14) & (divisor >= 24**2)
    cols = np.arange(x0, x1)
    top = valid.argmax(axis=0)
    bottom = len(valid) - valid[::-1].argmax(axis=0) - 1
    radius = (xs.stop - xs.start) / 2.
    eligible = (np.abs(cols - center[0]) >= max(3, .2 * radius))
    eligible &= np.abs(cols - center[0]) < .79 * radius
    eligible &= (bottom - top >= 1) & valid.any(axis=0) & mask[region].any(axis=0)
    indices = np.flatnonzero(eligible)
    upper = top[indices] + .5 - np.clip(alpha[top[indices], indices], 0., 1.)
    lower = bottom[indices] - .5 + np.clip(alpha[bottom[indices], indices], 0., 1.)
    return np.column_stack((cols[indices], .5 * (lower - upper)))


_PITCHES = np.linspace(-.85, .95, 61)
_AXIAL = np.column_stack((np.zeros_like(_PITCHES), np.cos(_PITCHES), np.sin(_PITCHES)))
_RADIAL = np.column_stack((np.zeros_like(_PITCHES), -np.sin(_PITCHES), np.cos(_PITCHES)))


def _profile_pitch(stereo, point, faces, fallback):
    residuals = []
    for camera, face in zip(stereo.rig, faces):
        if len(face.profile) < 3:
            continue
        cx, cy, depth = camera.project(point)
        right, up, forward = camera.basis
        sx, sy = (cx - camera.center[0]) / camera.focal, (camera.center[1] - cy) / camera.focal
        derivative = camera.focal / depth * np.array((right - sx * forward, sy * forward - up))
        radial = PADDLE_RADIUS * (_RADIAL @ derivative.T)
        axial = .006 * (_AXIAL @ derivative.T)
        var_x = (PADDLE_RADIUS * derivative[0, 0])**2 + radial[:, 0]**2
        slope = (radial[:, 0] * radial[:, 1]) / var_x
        radius_y = np.sqrt(np.maximum(0., radial[:, 1]**2 - slope**2 * var_x))
        offset = face.profile[:, 0] - cx
        shifted = offset - np.array((-1., 1.))[:, None, None] * axial[:, 0, None]
        half = radius_y[:, None] * np.sqrt(np.maximum(0., 1 - shifted**2 / var_x[:, None]))
        center = slope[:, None] * shifted + np.array((-1., 1.))[:, None, None] * axial[:, 1, None]
        height = .5 * ((center + half).max(axis=0) - (center - half).min(axis=0))
        residuals.append(np.abs(height - face.profile[:, 1]))
    if not residuals:
        return fallback
    errors = np.median(np.column_stack(residuals), axis=1)
    if fallback is not None:
        errors += .15 * np.abs(_PITCHES - fallback)
    best = np.argmin(errors)
    return float(_PITCHES[best]) if errors[best] < 1.1 else fallback


def _reconstruct_face(stereo, left, right, previous_pitch):
    top = max(left.top, right.top)
    bottom = min(left.top + len(left.middle), right.top + len(right.middle))
    vertical = .5 * (left.center[1] + right.center[1])
    center_left, center_right = left.center[0], right.center[0]
    pitch = previous_pitch
    if bottom - top >= 3:
        index = np.arange(top, bottom)
        li, ri = index - left.top, index - right.top
        widths = np.minimum(left.span[li], right.span[ri])
        valid = (widths > .48 * max(widths)) & (np.abs(left.span[li] - right.span[ri]) <= 4)
        if valid.sum() >= 3:
            offsets = index[valid] - vertical
            linear = np.column_stack((np.ones(len(offsets)), offsets))
            l = left.middle[li[valid]]
            r = right.middle[ri[valid]]
            fit_left = np.linalg.lstsq(linear, l, rcond=None)[0]
            fit_right = np.linalg.lstsq(linear, r, rcond=None)[0]
            center_left, center_right = fit_left[0], fit_right[0]
            disparity = center_left - center_right
            slope = fit_left[1] - fit_right[1]
            intercept = disparity + slope * (stereo.camera.center[1] - vertical)
            _, up, forward = stereo.camera.basis
            normal = intercept * forward - slope * stereo.focal * up
            candidate = np.arctan2(normal[2], normal[1])
            if abs(candidate) < 1.2 and valid.sum() >= 5:
                pitch = float(candidate)
    point = stereo.position((center_left, vertical), (center_right, vertical))
    if point is not None:
        pitch = _profile_pitch(stereo, point, (left, right), pitch)
    return point, pitch


class _Path:
    def __init__(self, identity, tick, position, area):
        self.identity = identity
        self.age = 0
        self.speed = np.zeros(3)
        self.position = position
        self.tick = tick
        self.observe(tick, position, area)

    def observe(self, tick, position, area):
        if self.age:
            speed = (position - self.position) / (tick - self.tick)
            self.speed = speed if self.age < 3 else self.speed * .35 + speed * .65
        self.tick = tick
        self.position = position
        self.area = area
        self.age += 1

    def predict(self, tick):
        return self.position + (tick - self.tick) * self.speed

    def track(self, tick):
        return Track(self.identity, tuple(self.predict(tick)), tuple(self.speed),
                     tick == self.tick, self.age)


class HallTracker:
    """Reconstruct compact moving regions and the responsive near racket."""

    def __init__(self, stereo, background, apertures, *, motion_supported_blade=False):
        self.stereo = stereo
        self.reference = background
        self.apertures = apertures
        self.motion_supported_blade = motion_supported_blade
        self.reset()

    def reset(self):
        self.tick = -1
        self.previous = None
        self.actor = self.ball = None
        self.actor_pitch = self.ball_sigma = None
        self.appearances = {}
        self.support = {}
        self._actor_path = self._ball_path = None
        self._palette = None

    def _actor_observation(self, rgb, differences):
        if self._palette is None:
            palettes = tuple(_material(image, difference, aperture)
                             for image, difference, aperture in zip(rgb, differences, self.apertures))
            if any(palette is None for palette in palettes):
                return
            self._palette = palettes
        recent = None
        if self._actor_path is not None and self.tick - self._actor_path.tick <= 5:
            position = self._actor_path.predict(self.tick)
            projected = [camera.project(position) for camera in self.stereo.rig]
            recent = [(np.array((x, y)), self.stereo.focal * PADDLE_RADIUS / max(depth, .20))
                      for x, y, depth in projected]
        priors = (None, None) if recent is None else recent
        faces = tuple(_face(image, reference, palette, prior, aperture,
                            motion_supported=self.motion_supported_blade)
                      for image, reference, palette, prior, aperture in zip(
                          rgb, self.reference, self._palette, priors, self.apertures))
        if any(face is None for face in faces):
            return
        left, right = faces
        if abs(left.center[1] - right.center[1]) > 5:
            return
        point, pitch = _reconstruct_face(self.stereo, left, right, self.actor_pitch)
        if point is None or not (-2.06 < point[1] < -.87 and -.42 < point[2] < 1.25):
            return
        area = .5 * (left.area + right.area)
        self.actor_pitch = pitch
        if self._actor_path is None:
            self._actor_path = _Path(0, self.tick, point, area)
        else:
            self._actor_path.observe(self.tick, point, area)
        self.support[0] = (area, area)

    def _ball_matches(self, blobs):
        estimate = None
        if self._ball_path is not None and self.tick - self._ball_path.tick <= 6:
            estimate = self.stereo.pixels(self._ball_path.predict(self.tick))
        best = None
        for left in blobs[0]:
            for right in blobs[1]:
                vertical = abs(left.center[1] - right.center[1])
                if vertical > 1.25:
                    continue
                position = self.stereo.position(left.center, right.center)
                if position is None:
                    continue
                x, y, z = position
                if abs(x) > 1.4 or abs(y) > 2.35 or not -.14 < z < 1.35:
                    continue
                depth = (position - self.stereo.camera.eye) @ self.stereo.camera.basis[2]
                diameter = 2 * self.stereo.focal * BALL_RADIUS / depth
                area = np.pi * (diameter / 2) ** 2
                observed_area = (left.area + right.area) / 2
                if observed_area > area * 2.9 + 2 or observed_area < area * .19:
                    continue
                ratio = np.log((observed_area + 1) / (area + 1))
                score = vertical * .54 + .6 * abs(ratio)
                score += .7 * np.linalg.norm(left.appearance - right.appearance)
                score -= .65 * (left.change + right.change)
                if estimate is None:
                    score += .65 * (1.4 - y)
                else:
                    error = max(np.linalg.norm(left.center - estimate[0]),
                                np.linalg.norm(right.center - estimate[1]))
                    if error > 8 + 5 * (self.tick - self._ball_path.tick):
                        continue
                    score += .27 * error
                if best is None or score < best[0]:
                    best = score, position, observed_area, left, right
        return None if best is None else best[1:]

    def update(self, frames):
        raw = np.asarray(frames)
        if raw.dtype != np.uint8 or raw.shape != self.reference.shape:
            raise ValueError('expected two calibrated RGB uint8 images')
        self.tick += 1
        rgb = raw.astype(np.int16)
        differences = tuple(_contrast(image, reference) for image, reference in zip(rgb, self.reference))
        recent = ((None,) * len(rgb) if self.previous is None else
                  tuple(_contrast(image, previous) for image, previous in zip(rgb, self.previous)))
        blobs = tuple(_blobs(image, difference, motion)
                      for image, difference, motion in zip(rgb, differences, recent))
        match = self._ball_matches(blobs)
        if match is not None:
            position, area, left, right = match
            if self._ball_path is None or self.tick - self._ball_path.tick > 6:
                self._ball_path = _Path(1, self.tick, position, area)
            else:
                self._ball_path.observe(self.tick, position, area)
            depth = (position - self.stereo.camera.eye) @ self.stereo.camera.basis[2]
            disparity = left.center[0] - right.center[0]
            sigma = .26 + .25 * abs(left.center[1] - right.center[1])
            metric = sigma * depth / self.stereo.focal
            self.ball_sigma = (metric, np.sqrt(2.) * sigma * depth / disparity, metric * 1.6)
            self.support[1] = (area, area)
        self.ball = (self._ball_path.track(self.tick) if self._ball_path is not None
                     and self.tick - self._ball_path.tick <= 6 else None)
        self._actor_observation(rgb, differences)
        self.actor = (self._actor_path.track(self.tick) if self._actor_path is not None
                      and self.tick - self._actor_path.tick <= 6 else None)
        self.previous = rgb
        return tuple(track for track in (self.actor, self.ball) if track is not None)


def hall_tracker_factory(rig, probe_histories, *, motion_supported_blade=False):
    """Fit a static image from previously collected action and stereo RGB pairs."""
    images = np.asarray([frames for history in probe_histories for _, frames in history[::2]])
    reference = modal_image(images)
    stereo = _Stereo(rig)
    apertures = _response_apertures(probe_histories, images)

    def factory():
        return HallTracker(stereo, reference, apertures,
                           motion_supported_blade=motion_supported_blade)

    return factory


def motion_tracker_factory(rig, probe_histories):
    """Use probe-relative RGB novelty to separate similar-colored blade and backdrop."""
    return hall_tracker_factory(rig, probe_histories, motion_supported_blade=True)
