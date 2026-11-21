"""Refine the detected TT22 ball without changing its stereo association."""

import numpy as np

from .tt4_arena import mujoco
from .tt18_view import ImageColors, triangulate_views
from .tt19_view import _in_court
from .tt19_view_trial import CAMERAS
from .tt22_rally_trial import RallyArena


SCALE = 2
OFFSETS = np.array([(x, y) for x in np.linspace(-.65, .65, 11) for y in np.linspace(-.65, .65, 11)])


def disk_center(pixels):
    """Fit a circular silhouette to the occupied and vacant pixel centers."""
    pixels = np.asarray(pixels, dtype=int)
    center = pixels.mean(axis=0)
    low = pixels.min(axis=0) - 1
    span = pixels.max(axis=0) - low + 2
    if len(pixels) < 5 or max(span) > 2 * min(span) or len(pixels) < .3 * np.prod(span - 2):
        return center
    occupied = np.zeros((span[1], span[0]), dtype=bool)
    offset = pixels - low
    occupied[offset[:, 1], offset[:, 0]] = True
    y, x = np.indices(occupied.shape)
    exterior = np.column_stack((x[~occupied], y[~occupied])) + low
    candidates = center + OFFSETS
    inside = np.square(candidates[:, None] - pixels).sum(axis=2).max(axis=1)
    outside = np.square(candidates[:, None] - exterior).sum(axis=2).min(axis=1)
    gap = np.sqrt(inside) - np.sqrt(outside)
    # The feasible cells describe the uncertainty due to raster sampling.
    accepted = gap <= max(.055, float(gap.min()) + .055)
    return candidates[accepted].mean(axis=0) if gap.min() < .30 else center


def crop_center(rgb, blob):
    """Read only the native high-resolution crop around a coarse detection."""
    center = (np.asarray(blob.center) + .5) * SCALE - .5
    half = int(np.ptp(blob.pixels, axis=0).max() * SCALE / 2 + 5)
    low = np.maximum(0, np.floor(center - half)).astype(int)
    high = np.minimum(rgb.shape[1::-1], np.ceil(center + half + 1)).astype(int)
    x0, y0 = low
    x1, y1 = high
    patch = ImageColors(rgb[y0:y1, x0:x1])
    objects = patch.components(.065, tolerance=.044, minimum_chroma=61., min_area=3,
                               max_span=max(6, 2*half-1))
    chosen = min(objects, key=lambda b: np.linalg.norm(b.center + low - center), default=None)
    if chosen is None or np.linalg.norm(chosen.center + low - center) > 2.3:
        return np.asarray(blob.center)
    fine = disk_center(chosen.pixels) + low
    return (fine + .5) / SCALE - .5


def position(tracker, mode, high=None):
    ball = tracker.ball
    if ball is None or not ball.observed or tracker.sources.get('ball_range') != 'stereo':
        return None, {}
    stock = np.asarray(ball.position)
    blobs = tracker.detections['ball']
    if mode == 'stock':
        return stock, {eye: blob.center for eye, blob in blobs.items()}
    if mode == 'disk':
        centers = {eye: disk_center(blob.pixels) for eye, blob in blobs.items()}
    else:
        centers = {eye: crop_center(high[eye], blob) for eye, blob in blobs.items()}
    refined = triangulate_views(tracker.fits, centers)
    return (refined if _in_court('ball', refined) else stock), centers


class DetailArena(RallyArena):
    """A second native sensor resolution with the same optical centers and FOV."""

    def __init__(self, *args):
        self.detail_renderer = None
        super().__init__(*args)

    def render_detail(self):
        self._color_roles()
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        if self.detail_renderer is None:
            settings, quality = self.model.vis.global_, self.model.vis.quality
            previous = (settings.offwidth, settings.offheight, quality.offsamples)
            try:
                settings.offwidth = self.width * SCALE
                settings.offheight = self.height * SCALE
                quality.offsamples = 0
                self.detail_renderer = mujoco.Renderer(self.model, height=settings.offheight, width=settings.offwidth)
            finally:
                settings.offwidth, settings.offheight, quality.offsamples = previous
        return {eye: self._render_view(self.detail_renderer, self.data, CAMERAS[eye][0]) for eye in self.streams}

    def close(self):
        if self.detail_renderer is not None:
            self.detail_renderer.close()
            self.detail_renderer = None
        super().close()
