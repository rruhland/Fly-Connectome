"""Metric RGB observations for table play with the SO-101 wrist camera."""

from hashlib import sha256

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .tt4_arena import BALL_RADIUS
from .tt5_vision import _Path
from .tt18_view import (CameraFit, ImageColors, LANDMARKS as HALL_LANDMARKS, Landmark, OBJECT_HUES, ViewRegistry,
                        _landmark_pixels, _matrix_seed, triangulate_views)
from .tt18_viewplay import _position_sigma, camera_state


WRIST = 'wrist'
WRIST_FOV = 93.
STREAMS = {'wrist+fixed': (WRIST, 'left', 'right'), 'wrist+side': (WRIST, 'side'),
           'wrist': (WRIST,), 'fixed': ('left', 'right')}
MODES = tuple(STREAMS)
# The markers are small paint-colored, noncolliding visual geometry on the rim.
# The coordinates describe the visible racket, independent of its wrist mount.
RIM_HUES = (.943, .594, .024, .404)
RIM_RADIUS = .080
RIM_ANGLES = np.deg2rad((23., 108., 194., 289.))
RIM = RIM_RADIUS * np.column_stack((np.cos(RIM_ANGLES), np.sin(RIM_ANGLES), np.zeros(4)))
TIP_HUE = .271
LANDMARKS = tuple(Landmark(position, marker.hue) for position, marker in zip((
    (-.61, -.49, .36), (.57, -.46, .56), (-.64, -.24, .19), (.61, -.19, .27),
    (-.59, .04, .43), (.59, .06, .50), (-.63, .34, .28), (.60, .31, .14),
    (-.66, .62, .17), (.66, .62, .38), (-.62, .95, .48), (.64, .95, .25),
), HALL_LANDMARKS, strict=True))


def _blob(colors, hue, *, tolerance=.018, minimum=77., area=2, span=None, around=None, border=False):
    maximum_span = colors.shape[0] if span is None else span
    candidates = tuple(colors.components(hue, tolerance=tolerance, minimum_chroma=minimum, min_area=area,
                                         max_span=maximum_span, border=border))
    if around is not None:
        candidates = tuple(blob for blob in candidates
                           if np.linalg.norm(blob.center - around) < .46 * colors.shape[0])
    return max(candidates, key=lambda b: (b.weights.max(), b.weights.sum()), default=None)


def detect_roles(images):
    result = {role: {} for role in ('ball', 'blade', 'tip', *range(len(RIM)))}
    for stream, colors in images.items():
        height = colors.shape[0]
        ball = _blob(colors, OBJECT_HUES['ball'], tolerance=.030, minimum=73., area=1, span=max(7, height // 9))
        blade = _blob(colors, OBJECT_HUES['blade'], tolerance=.031, minimum=74., area=3, border=True)
        tip = _blob(colors, TIP_HUE, tolerance=.016, minimum=82., area=2, span=max(10, height // 3))
        for role, blob in (('ball', ball), ('blade', blade), ('tip', tip)):
            if blob is not None:
                result[role][stream] = blob
        if blade is not None:
            for role, hue in enumerate(RIM_HUES):
                mark = _blob(colors, hue, around=blade.center, tolerance=.016, minimum=85.,
                             span=max(10, height // 4))
                if mark is not None:
                    result[role][stream] = mark
    return result


def _rgb_rim(cameras, detections):
    model, world = [], []
    for index, point in enumerate(RIM):
        seen = {key: blob.center for key, blob in detections[index].items()}
        metric = triangulate_views(cameras, seen)
        if metric is not None:
            model.append(point)
            world.append(metric)
    if len(world) < 3:
        return None
    local = np.asarray(model)
    observed = np.asarray(world)
    center = local.mean(axis=0)
    midpoint = observed.mean(axis=0)
    left, _, right = np.linalg.svd((local - center).T @ (observed - midpoint))
    left[:, -1] *= np.linalg.det(left @ right)
    rotation = left @ right
    position = midpoint - center @ rotation
    if np.max(np.linalg.norm(local @ rotation + position - observed, axis=1)) > .013:
        return None
    return position, rotation[2]


def _image_rim(camera, pixels, *, refine=False):
    """Recover a known planar marker pose in metric camera coordinates."""
    xyz = RIM / RIM_RADIUS
    uv = (pixels - camera.center) / camera.focal
    plane = xyz.copy()
    plane[:, -1] = 1
    zeros = np.zeros_like(plane)
    design = np.concatenate((np.column_stack((plane, zeros, -uv[:, :1] * plane)),
                             np.column_stack((zeros, plane, -uv[:, 1:] * plane))))
    _, singular, vh = np.linalg.svd(design)
    if singular[-1] < .005 * singular[0]:
        return None
    homography = vh[-1].reshape(3, 3)
    scale = np.sign(homography[-1, -1]) * np.mean(np.linalg.norm(homography[:, :2], axis=0))
    if abs(scale) < 1e-7:
        return None
    first, second, translation = (homography / scale).T
    left, _, right = np.linalg.svd(np.column_stack((first, second, np.cross(first, second))))
    rotation = left @ right
    translation *= RIM_RADIUS
    if not .039 < translation[2] < 4.0:
        return None

    if refine:
        # Nearby rim paint is partially hidden by the blade. A perspective
        # rigid fit averages the resulting unequal centroid shifts.
        def residual(parameters):
            local_rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
            local = RIM @ local_rotation.T + parameters[3:]
            projection = camera.center + camera.focal * local[:, :2] / np.maximum(.01, local[:, 2:])
            return (projection - pixels).ravel()

        start = np.r_[Rotation.from_matrix(rotation).as_rotvec(), translation]
        if np.max(np.abs(residual(start))) > 2.:
            solution = least_squares(residual, start, loss='soft_l1', max_nfev=15)
            rotation = Rotation.from_rotvec(solution.x[:3]).as_matrix()
            translation = solution.x[3:]
    point = camera.eye + translation @ camera.rotation
    projection = camera.project(point + RIM @ rotation.T @ camera.rotation)
    error = np.linalg.norm(projection[:, :2] - pixels, axis=1)
    if np.min(projection[:, 2]) <= .02 or np.max(error) > 2.4:
        return None
    return point, rotation[:, 2] @ camera.rotation


def _mono_rim(cameras, detections, *, refine=False):
    for stream in (WRIST, *[key for key in cameras if key != WRIST]):
        if stream not in cameras or any(stream not in detections[index] for index in range(len(RIM))):
            continue
        pixels = np.stack([detections[index][stream].center for index in range(len(RIM))])
        pose = _image_rim(cameras[stream], pixels, refine=refine and stream == WRIST)
        if pose is not None:
            return pose, stream
    return None, None


def _mono_ball(camera, blob):
    """Read range from a resolved sphere and bound the rasterization contribution."""
    span = np.ptp(blob.pixels, axis=0) + 1
    radius = .5 * min(span)
    if radius < 3. or max(span) > 1.38 * min(span):
        return None
    ray = camera.ray(blob.center)
    obliquity = np.linalg.norm(ray)
    distance = camera.focal * BALL_RADIUS * obliquity / radius
    width = .65 * distance / radius
    if width > .13:
        return None
    return camera.eye + distance / obliquity * ray, (max(.018, width),) * 3


def _triangulate(cameras, blobs):
    pixels = {key: blob.center for key, blob in blobs.items()}
    point = triangulate_views(cameras, pixels)
    if point is None:
        return None, ()
    streams = tuple(key for key in cameras if key in pixels
                    and np.linalg.norm(cameras[key].project(point)[:2] - pixels[key]) <= 2.)
    return point, streams


def _in_court(role, point):
    if point is None or not np.isfinite(point).all():
        return False
    x, y, z = point
    return (abs(x) < 1.5 and abs(y) < 2.5 and -.19 < z < 1.55) if role == 'ball' else (
        abs(x) < 1.5 and -2.3 < y < -.20 and -.53 < z < 1.40)


def _wrist_fit(image, prior, fovy=WRIST_FOV):
    observations = tuple(_landmark_pixels(image, LANDMARKS))
    if len(observations) < (6 if prior is None else 5):
        return None, 'insufficient_landmarks'
    world, pixels = map(np.asarray, zip(*observations, strict=True))
    height, width = image.shape
    focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))

    def view(parameters):
        rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
        return CameraFit(parameters[3:6], rotation, focal, width, height)

    def residual(parameters):
        return (view(parameters).project(world)[:, :2] - pixels).ravel()

    if prior is not None:
        errors = np.linalg.norm(prior.project(world)[:, :2] - pixels, axis=1)
        if np.count_nonzero(errors < .45) >= len(world) - 1:
            return prior, 'verified'
        starts = [np.r_[Rotation.from_matrix(prior.rotation).as_rotvec(), prior.eye]]
    else:
        starts = []
    initial = _matrix_seed(world, pixels, image.shape)
    if initial is not None:
        starts.append(initial[:6])
    for start in starts:
        if min(view(start).project(world)[:, 2]) < .08:
            continue
        fit = least_squares(residual, start, loss='soft_l1', f_scale=.4, max_nfev=13)
        candidate = view(fit.x)
        projected = candidate.project(world)
        errors = np.linalg.norm(projected[:, :2] - pixels, axis=1)
        valid = (errors < 1.25) & (projected[:, 2] > .08)
        if np.count_nonzero(valid) >= max(5, len(world) - 2):
            rms = float(np.sqrt(np.mean(errors[valid]**2)))
            if rms < .80:
                return CameraFit(candidate.eye, candidate.rotation, focal, width, height, rms,
                                 int(np.count_nonzero(valid))), 'relocated' if prior is not None else 'added'
    return None, 'rejected'


class ViewTransport:
    """Hold persistent RGB camera fits while the player resets per-point tracks."""

    def __init__(self, mode='wrist+fixed', state=None, *, monocular_ball=False, wrist_fovy=WRIST_FOV):
        if mode not in MODES:
            raise ValueError(f'unknown SO-101 camera mode {mode!r}')
        self.mode = mode
        self.monocular_ball = bool(monocular_ball or mode == 'wrist')
        self.wrist_fovy = wrist_fovy
        self.registry = ViewRegistry(LANDMARKS)
        if state is not None:
            self.registry.known.update({key: CameraFit(**dict(fit, eye=np.asarray(fit['eye']),
                                                            rotation=np.asarray(fit['rotation'])))
                                        for key, fit in state.items()})

    def tracker(self):
        return SOViewTracker(self)

    def observe(self, images, tick, previous):
        fixed = {key: value for key, value in images.items() if key != WRIST}
        registry = self.registry
        if tick % 8 == 0 or set(fixed) != set(previous) - {WRIST}:
            fits = dict(registry.observe(fixed))
        else:
            fits = {key: previous[key] for key in fixed}
            registry.last_events = {key: 'verified' for key in fixed}
        if WRIST in images:
            fit, reason = _wrist_fit(images[WRIST], registry.known.get(WRIST), self.wrist_fovy)
            registry.last_events[WRIST] = reason
            if fit is not None:
                registry.known[WRIST] = fits[WRIST] = fit
        elif WRIST in previous:
            registry.last_events[WRIST] = 'removed'
        return fits

    def state(self):
        return camera_state(self.registry.known)


class SOViewTracker:
    """A tracker with the archived table player's RGB, track and uncertainty API."""

    def __init__(self, transport):
        self.transport = transport
        self.tick = -1
        self.actor = self.ball = self.actor_pitch = self.actor_yaw = self.actor_normal = self.ball_sigma = None
        self.appearances = {0: 10, 1: 1}
        self.support = {}
        self.sightings = []
        self.registration = []
        self.sources = {}
        self.detections = {}
        self.fits = {}
        self.paths = {}
        self.digest = sha256()

    def _track(self, role, position, detections):
        identity = 0 if role == 'blade' else 1
        area = np.mean([len(blob.pixels) for blob in detections.values()]) if detections else 1.
        if identity not in self.paths or self.tick - self.paths[identity].tick > 6:
            self.paths[identity] = _Path(identity, self.tick, position, area)
        else:
            self.paths[identity].observe(self.tick, position, area)
        self.support[identity] = (area, area)
        return self.paths[identity].track(self.tick)

    def update(self, frames):
        self.tick += 1
        images = {}
        for key, rgb in frames.items():
            if key not in STREAMS[self.transport.mode]:
                continue
            self.digest.update(key.encode())
            self.digest.update(rgb)
            images[key] = ImageColors(rgb)
        self.fits = self.transport.observe(images, self.tick, self.fits)
        changes = {key: event for key, event in self.transport.registry.last_events.items() if event != 'verified'}
        if changes or self.tick == 0:
            self.registration.append(dict(tick=self.tick, events=changes))
        self.detections = detections = detect_roles(images)
        self.actor = self.ball = self.actor_pitch = self.actor_yaw = self.actor_normal = self.ball_sigma = None
        self.support.clear()
        self.sources = {}
        blade, streams = _triangulate(self.fits, detections['blade'])
        pose = _rgb_rim(self.fits, detections)
        pose_source = 'multiple_views' if pose is not None else None
        if pose is None:
            pose, pose_source = _mono_rim(self.fits, detections, refine=self.transport.mode == 'wrist')
        if pose is not None:
            position, normal = pose
            if _in_court('blade', position) and (blade is None or np.linalg.norm(blade - position) < .07):
                blade = position
                self.actor_normal = normal if normal[1] >= 0. else -normal
                self.actor_pitch = float(np.arctan2(self.actor_normal[2], np.linalg.norm(self.actor_normal[:2])))
                self.actor_yaw = float(np.arctan2(self.actor_normal[0], self.actor_normal[1]))
                self.sources['blade_pose'] = pose_source
        if _in_court('blade', blade):
            self.actor = self._track('blade', blade, detections['blade'])
            self.sources['blade'] = streams
        ball, streams = _triangulate(self.fits, detections['ball'])
        if ball is not None:
            self.ball_sigma = _position_sigma(ball, self.fits, detections['ball'])
        elif self.transport.monocular_ball and WRIST in self.fits and WRIST in detections['ball']:
            mono = _mono_ball(self.fits[WRIST], detections['ball'][WRIST])
            if mono is not None:
                ball, self.ball_sigma = mono
                self.sources['ball_range'] = 'known_ball_size'
                streams = (WRIST,)
        if _in_court('ball', ball):
            self.ball = self._track('ball', ball, detections['ball'])
            self.sources['ball'] = streams
        self.sightings.append([len(self.fits), self.ball is not None, self.actor is not None,
                               self.actor_pitch is not None, WRIST in detections['tip'],
                               WRIST in detections['blade'], WRIST in self.fits])
        return tuple(track for track in (self.actor, self.ball) if track is not None)
