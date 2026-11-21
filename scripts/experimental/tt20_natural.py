"""Table-relative RGB tracks from normal court lines, ball and blade silhouettes."""

from collections import Counter
from types import SimpleNamespace

import numpy as np

from .tt4_arena import BALL_RADIUS
from .tt18_view import CameraFit, ImageColors
from .tt18_viewplay import _position_sigma, camera_state
from .tt19_view import SOViewTracker, _blob, _in_court, _triangulate
from .tt20_mono_flight import MonoFlight
from .tt20_natural_association import choose_component
from .tt20_natural_geometry import register_table


_PUBLIC_FOV = dict(wrist=84., left=57., right=57., side=62., broadcast=55.)
_STREAMS = {'wrist+fixed': ('wrist', 'left', 'right'), 'wrist+side': ('wrist', 'side'),
            'fixed': ('left', 'right'), 'wrist': ('wrist',), 'left': ('left',), 'right': ('right',),
            'outside': ('left',), 'left-only': ('left',), 'right-only': ('right',), 'side': ('side',)}
_BLADE_RADIUS = .076


def _directions(camera, pixels):
    value = np.column_stack(((np.asarray(pixels) - camera.center) / camera.focal, np.ones(len(pixels))))
    return value / np.linalg.norm(value, axis=1, keepdims=True)


def _size_ball(camera, blob):
    points = np.asarray(blob.pixels, dtype=int)
    if len(points) < 5:
        return None
    low, high = points.min(axis=0), points.max(axis=0)
    span = high - low + 1
    if min(span) < 2 or max(span) > 1.8 * min(span) or len(points) < .39 * np.prod(span):
        return None
    gx, gy = np.meshgrid(np.arange(low[0] - 1, high[0] + 2), np.arange(low[1] - 1, high[1] + 2))
    window = np.column_stack((gx.ravel(), gy.ravel()))
    mask = np.zeros(gx.shape, dtype=bool)
    mask[points[:, 1] - low[1] + 1, points[:, 0] - low[0] + 1] = True
    center = points.mean(axis=0)
    dx, dy = np.meshgrid(np.linspace(-.55, .55, 7), np.linspace(-.55, .55, 7))
    candidates = center + np.column_stack((dx.ravel(), dy.ravel()))
    directions = _directions(camera, candidates)
    rays = _directions(camera, window)
    angular = np.sqrt(np.maximum(0., 1. - (directions @ rays.T) ** 2))
    # For a sampled circular silhouette the farthest inside ray and the
    # closest outside ray bracket the angular radius. Allow slight raster,
    # tessellation and color-threshold disagreement.
    lower = angular[:, mask.ravel()].max(axis=1)
    upper = angular[:, ~mask.ravel()].min(axis=1)
    offset = np.linalg.norm(candidates - center, axis=1)
    loss = 3. * np.maximum(0., lower - upper) + .12 * offset / camera.focal
    if camera.focal * min(loss) > 1.15:
        return None
    fitted = int(np.argmin(loss))
    alpha = .5 * (lower[fitted] + upper[fitted])
    if not .0008 < alpha < .50:
        return None
    distance = BALL_RADIUS / alpha
    if not .047 < distance < 6.2:
        return None
    direction = directions[fitted] @ camera.rotation
    near = loss <= loss[fitted] + .18 / camera.focal
    sizes = .5 * (lower[near] + upper[near])
    radial = max(.012, distance * (.24 / (camera.focal * alpha) + np.std(sizes) / alpha))
    transverse = max(.005, .55 * distance / camera.focal)
    sigma = np.sqrt((radial * direction) ** 2 + transverse ** 2 * (1. - direction ** 2))
    return camera.eye + distance * direction, tuple(sigma)


def _blade_shape(camera, blob):
    if len(blob.pixels) < 16:
        return None
    rays = _directions(camera, blob.pixels)
    direction = rays.mean(axis=0)
    direction /= np.linalg.norm(direction)
    residual = rays - direction
    _, sizes, axes = np.linalg.svd(residual, full_matrices=False)
    alpha = 2. * sizes[0] / np.sqrt(len(rays))
    if not .003 < alpha < .65:
        return None
    fraction = sizes[1] / sizes[0]
    if fraction < .16:
        return None
    position = camera.eye + (_BLADE_RADIUS / alpha) * (direction @ camera.rotation)
    across = axes[1] - (axes[1] @ direction) * direction
    across /= np.linalg.norm(across)
    normal = fraction * direction
    swing = np.sqrt(max(0., 1. - fraction ** 2)) * across
    candidates = [(normal + sign * swing) @ camera.rotation for sign in (-1., 1.)]
    facing = [vector if vector[1] >= 0. else -vector for vector in candidates]
    return position, facing


def _blade(cameras, detections):
    position, streams = _triangulate(cameras, detections)
    shapes = {key: _blade_shape(cameras[key], blob) for key, blob in detections.items() if key in cameras}
    shapes = {key: result for key, result in shapes.items() if result is not None}
    if position is None and len(shapes) == 1:
        stream, result = next(iter(shapes.items()))
        return result[0], (stream,), None
    if position is None or len(shapes) < 2:
        return position, streams, None
    pairings = [(float(first @ second), first, second)
                for index, first_set in enumerate(shapes.values())
                for second_set in list(shapes.values())[index + 1:]
                for first in first_set[1] for second in second_set[1]]
    agreement, first, second = max(pairings, key=lambda pair: pair[0])
    normal = first + second
    return position, streams, normal / np.linalg.norm(normal) if agreement > .95 else None


class NaturalViewTransport:
    """Persist only camera calibrations inferred from native lines, not world poses."""

    def __init__(self, mode='wrist+fixed', state=None, *, lens_fov=None, flight=True, law=None,
                 bounds=(.424, .80), continuous_wrist=False, frame_seconds=1. / 30., associate=False):
        if mode not in _STREAMS:
            raise ValueError(f'unknown natural view mode {mode!r}')
        self.mode = mode
        self.bounds = bounds
        self.lenses = _PUBLIC_FOV if lens_fov is None else lens_fov
        self.flight = bool(flight)
        self.law = law
        self.continuous_wrist = bool(continuous_wrist)
        self.associate = bool(associate)
        self.frame_seconds = frame_seconds
        known = {} if state is None else {
            key: CameraFit(**dict(fit, eye=np.asarray(fit['eye']), rotation=np.asarray(fit['rotation'])))
            for key, fit in state.items()}
        self.registry = SimpleNamespace(known=known, active={}, last_events={}, counts=Counter())

    def tracker(self):
        return NaturalViewTracker(self)

    def focal(self, stream, rgb):
        spec = self.lenses.get(stream, _PUBLIC_FOV[stream])
        degrees = spec['fovy'] if isinstance(spec, dict) else spec
        return rgb.shape[0] / (2. * np.tan(np.deg2rad(degrees) / 2.))

    def observe(self, frames, tick, prior, wrist_prior=None):
        fits = {}
        events = {stream: 'removed' for stream in prior.keys() - frames.keys()}
        known = self.registry.known
        for stream, rgb in frames.items():
            focal = self.focal(stream, rgb)
            previous = known.get(stream)
            if (stream != 'wrist' and stream in prior and tick % 8 != 0
                    and abs(prior[stream].focal - focal) < .01):
                fits[stream] = prior[stream]
                events[stream] = 'verified'
                continue
            motion = None
            if stream == 'wrist' and wrist_prior is not None and self.continuous_wrist:
                camera, earlier_tick = wrist_prior
                motion = camera, (tick - earlier_tick) * self.frame_seconds * 30.
            fit = register_table(rgb, self.bounds, focal, previous, motion=motion,
                                 search=stream != 'wrist' or previous is None or tick % 4 == 0)
            if fit is None:
                events[stream] = 'insufficient_court_lines'
                continue
            fits[stream] = known[stream] = fit
            events[stream] = ('verified' if fit is previous else 'tracked' if previous is not None else 'added')
        self.registry.active = fits
        self.registry.last_events = events
        self.registry.counts.update(events.values())
        return fits

    def state(self):
        return camera_state(self.registry.known)


class NaturalViewTracker(SOViewTracker):
    def __init__(self, transport):
        super().__init__(transport)
        self.raw_ball = self.raw_ball_sigma = None
        self.ball_velocity = None
        self._wrist_prior = None
        self.flight = (MonoFlight(bounds=transport.bounds, law=transport.law, frame_seconds=transport.frame_seconds)
                       if transport.flight else None)

    def update(self, frames, *, tick=None):
        if tick is not None and tick <= self.tick:
            raise ValueError('natural view frames must arrive in time order')
        self.tick = self.tick + 1 if tick is None else tick
        frames = {key: rgb for key, rgb in frames.items() if key in _STREAMS[self.transport.mode]}
        detections = {name: {} for name in ('ball', 'blade', 'tip')}
        self.fits = self.transport.observe(frames, self.tick, self.fits, self._wrist_prior)
        if 'wrist' in self.fits:
            self._wrist_prior = self.fits['wrist'], self.tick
        for key, rgb in frames.items():
            self.digest.update(key.encode())
            self.digest.update(rgb)
            colors = ImageColors(rgb)
            ball = _blob(colors, .065, tolerance=.044, minimum=61., area=1,
                         span=max(10, colors.shape[0] // 3))
            blade = _blob(colors, .993, tolerance=.027, minimum=57., area=4)
            if blade is None:
                blade = _blob(colors, .878, tolerance=.030, minimum=68., area=4)
            for identity, label, blob, fitter in ((1, 'ball', ball, _size_ball), (0, 'blade', blade, _blade_shape)):
                if self.transport.associate:
                    blob = choose_component(colors, label, self.fits.get(key), fitter, self.transport.bounds,
                                            self.paths.get(identity), self.tick, self.transport.frame_seconds, blob)
                if blob is not None:
                    detections[label][key] = blob
        events = {key: event for key, event in self.transport.registry.last_events.items() if event != 'verified'}
        if events or self.tick == 0:
            self.registration.append(dict(tick=self.tick, events=events))
        self.detections = detections
        self.actor = self.ball = self.actor_pitch = self.actor_yaw = self.actor_normal = self.ball_sigma = None
        self.raw_ball = self.raw_ball_sigma = None
        self.ball_velocity = None
        self.support.clear()
        self.sources = {}
        blade, streams, normal = _blade(self.fits, detections['blade'])
        if _in_court('blade', blade):
            self.actor = self._track('blade', blade, detections['blade'])
            self.sources['blade'] = streams
            if normal is not None:
                self.actor_normal = normal
                self.actor_pitch = float(np.arctan2(normal[2], np.linalg.norm(normal[:2])))
                self.actor_yaw = float(np.arctan2(normal[0], normal[1]))
                self.sources['blade_pose'] = 'two_ellipses'
        balls = detections['ball']
        ball, streams = _triangulate(self.fits, balls)
        if ball is not None:
            self.ball_sigma = _position_sigma(ball, self.fits, balls)
            self.sources['ball_range'] = 'stereo'
        if ball is None:
            observed = [key for key in self.fits if key in balls]
            stream = max(observed, key=lambda key: len(balls[key].pixels), default=None)
            camera, blob = (self.fits[stream], balls[stream]) if stream is not None else (None, None)
            raw = None if camera is None else _size_ball(camera, blob)
            if raw is not None and _in_court('ball', raw[0]):
                self.raw_ball, self.raw_ball_sigma = raw
                ball, self.ball_sigma = raw
                self.sources['ball_range'] = 'size'
            if self.flight is not None:
                fit = self.flight.observe(self.tick, camera, blob, self.raw_ball, size_sigma=self.raw_ball_sigma)
                if fit is not None and _in_court('ball', fit[0]):
                    ball, self.ball_sigma = fit
                    self.ball_velocity = self.flight.velocity
                    self.sources['ball_range'] = self.flight.source
            streams = (stream,) if stream is not None else ()
        if _in_court('ball', ball):
            self.ball = self._track('ball', ball, balls)
            self.sources['ball'] = streams
        self.sightings.append([len(self.fits), self.ball is not None, self.actor is not None,
                               self.actor_pitch is not None, False, 'wrist' in detections['blade'],
                               'wrist' in self.fits])
        return tuple(track for track in (self.actor, self.ball) if track is not None)
