"""Feed the archived table controller metric tracks derived from current RGB streams."""

from hashlib import sha256

import numpy as np

from .tt5_vision import _Path
from .tt18_view import CameraFit, ImageColors, ViewRegistry, object_blobs, triangulate_views


POSE_HUE = .943
POSE_RADIUS = .083


def camera_state(cameras):
    return {key: dict(eye=fit.eye.tolist(), rotation=fit.rotation.tolist(), focal=fit.focal,
                      width=fit.width, height=fit.height, rms=fit.rms, markers=fit.markers)
            for key, fit in cameras.items()}


class ViewTransport:
    """Persist frame registration separately from the point-reset rally tracker."""

    def __init__(self, adaptive, state=None):
        self.adaptive = adaptive
        self.registry = ViewRegistry()
        if state is not None:
            self.registry.known.update({key: CameraFit(**dict(fit, eye=np.array(fit['eye']),
                                                            rotation=np.array(fit['rotation'])))
                                        for key, fit in state.items()})

    def tracker(self):
        return RallyViewTracker(self)

    def register(self, images):
        if self.adaptive or not self.registry.known:
            return self.registry.observe(images)
        self.registry.last_events = {key: 'frozen' if key in self.registry.known else 'unknown_frozen'
                                     for key in images}
        return {key: self.registry.known[key] for key in images if key in self.registry.known}

    def state(self):
        return camera_state(self.registry.known)


def blade_pitch(images, blade_views, fits, position):
    """Read the known radial direction from the blade center to a painted rim marker."""
    pixels = {}
    for stream in fits:
        if stream not in blade_views:
            continue
        colors = images[stream]
        center = blade_views[stream].center
        nearby = [blob for blob in colors.components(POSE_HUE, tolerance=.017, minimum_chroma=88.,
                                                      min_area=2, max_span=20)
                  if np.linalg.norm(blob.center - center) < max(12, colors.shape[0] / 8)]
        if nearby:
            pixels[stream] = min(nearby, key=lambda blob: np.linalg.norm(blob.center - center)).center
    cue = triangulate_views(fits, pixels)
    if cue is None:
        return None
    axis = cue - position
    pitch = np.arctan2(-axis[1], axis[2])
    if abs(np.linalg.norm(axis) - POSE_RADIUS) > .023 or not -.84 < pitch < 1.01:
        return None
    return float(pitch)


def _position_sigma(point, fits, blobs):
    rows = []
    for stream in fits:
        if stream not in blobs:
            continue
        camera = fits[stream]
        u, v, depth = camera.project(point)
        if depth > 0 and np.linalg.norm((u, v) - blobs[stream].center) < 2:
            fractional = (np.array((u, v)) - camera.center) / camera.focal
            rows.extend(camera.focal / depth * (camera.rotation[:2] - fractional[:, None] * camera.rotation[2]))
    if len(rows) < 4:
        return None
    covariance = .34**2 * np.linalg.pinv(np.asarray(rows).T @ rows)
    return tuple(np.maximum((.004, .007, .004), np.sqrt(np.diag(covariance))))


class RallyViewTracker:
    """Expose the same camera-track interface as the archived stereo tracker."""

    def __init__(self, transport):
        self.transport = transport
        self.tick = -1
        self.support = {}
        self.appearances = {0: 10, 1: 1}
        self.actor = self.ball = self.actor_pitch = self.ball_sigma = None
        self.paths = {}
        self.streams = set()
        self.fits = {}
        self.registration = []
        self.sightings = []
        self.digest = sha256()

    def update(self, frames):
        self.tick += 1
        images = {}
        for key, rgb in frames.items():
            self.digest.update(key.encode())
            self.digest.update(rgb)
            images[key] = ImageColors(rgb)
        if self.tick == 0 or set(frames) != self.streams or self.tick % 8 == 0:
            self.fits = self.transport.register(images)
            self.registration.append(dict(tick=self.tick, events=dict(self.transport.registry.last_events)))
        self.streams = set(frames)
        self.ball_sigma = self.actor_pitch = None
        self.support.clear()
        self.actor = self.ball = None
        if len(self.fits) >= 2:
            blobs = object_blobs(images)
            for identity, role in enumerate(('blade', 'ball')):
                candidates = blobs[role]
                position = triangulate_views(self.fits, {key: blob.center for key, blob in candidates.items()})
                if position is None:
                    continue
                x, y, z = position
                if role == 'blade' and (abs(x) > 1.0 or not -2.10 < y < -.94 or not -.42 < z < 1.25):
                    continue
                if role == 'ball' and (abs(x) > 1.48 or abs(y) > 2.47 or not -.20 < z < 1.42):
                    continue
                area = np.mean([len(blob.pixels) for blob in candidates.values()])
                if identity not in self.paths or self.tick - self.paths[identity].tick > 6:
                    self.paths[identity] = _Path(identity, self.tick, position, area)
                else:
                    self.paths[identity].observe(self.tick, position, area)
                self.support[identity] = (area, area)
                track = self.paths[identity].track(self.tick)
                if role == 'blade':
                    self.actor = track
                    self.actor_pitch = blade_pitch(images, candidates, self.fits, position)
                else:
                    self.ball = track
                    self.ball_sigma = _position_sigma(position, self.fits, candidates)
        self.sightings.append([len(self.fits), self.ball is not None, self.actor is not None,
                               self.actor_pitch is not None])
        return tuple(track for track in (self.actor, self.ball) if track is not None)
