"""Render RGB observations for a carriage and a delayed three-joint position arm."""

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from body_reaching_arena import GOAL_TOLERANCE, SUCCESS_TICKS, TICK_SECONDS
from planar_position_arm import (
    _configuration_for,
    _joint_points,
    _relative_setpoint,
)


RGB_CAMERA_SIZE = 160
MAX_COMMAND_DELAY = 3
TIP_COLOR = (255, 188, 66)
GOAL_COLOR = (235, 83, 183)
REFERENCE_COLOR = (96, 221, 237)
FIDUCIAL_COLORS = ((69, 227, 99), (96, 110, 248), (235, 217, 171))
PHYSICAL_RING_RADIUS = .056
OPTICAL_SCALE_RANGE = (69., 75.)


@dataclass(frozen=True)
class CameraView:
    """An overhead camera with a private, time-varying similarity transform."""

    rotation: float
    scale: float
    translation: tuple[float, float]
    phase: tuple[float, float, float]
    illumination: tuple[int, int, int]

    @classmethod
    def sample(cls, rng, *, changed=False):
        rotation = rng.uniform(.86, 2.22) * rng.choice((-1, 1)) if changed else rng.uniform(-.44, .44)
        scale = rng.uniform(*OPTICAL_SCALE_RANGE)
        translation = tuple(rng.uniform(-5., 5., 2))
        if changed:
            translation = tuple(value + shift for value, shift in zip(translation, (4., -4.)))
        return cls(rotation, scale, translation, tuple(rng.uniform(-np.pi, np.pi, 3)),
                   tuple(map(int, rng.integers(-6, 7, 3))))

    def at(self, tick):
        """Return a world-to-pixel matrix and origin for the renderer and evaluator."""
        px, py, pr = self.phase
        yaw = self.rotation + .006 * math.sin(.49 * tick + pr)
        scale = self.scale * (1. + .004 * math.sin(.67 * tick + px))
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        matrix = scale * np.array(((cos_yaw, -sin_yaw), (-sin_yaw, -cos_yaw)))
        center = (RGB_CAMERA_SIZE - 1) / 2
        origin = np.array((center + self.translation[0] + 1.4 * math.sin(.31 * tick + px),
                           center + self.translation[1] + 1.4 * math.sin(.37 * tick + py)))
        return matrix, origin


_ROWS, _COLUMNS = np.indices((RGB_CAMERA_SIZE, RGB_CAMERA_SIZE))
_BACKGROUND = np.stack((25 + _COLUMNS // 40, 31 + _ROWS // 31, 37 + (_ROWS + _COLUMNS) // 31), axis=-1)
_BACKGROUND += (((_ROWS * 13) ^ (_COLUMNS * 7)) % 5)[..., None]
_BACKGROUND = _BACKGROUND.astype(np.uint8)


def _circle(image, center, radius, color, *, inner_radius=0.):
    x, y = center
    top, bottom = max(0, math.floor(y - radius)), min(RGB_CAMERA_SIZE, math.ceil(y + radius) + 1)
    left, right = max(0, math.floor(x - radius)), min(RGB_CAMERA_SIZE, math.ceil(x + radius) + 1)
    if top >= bottom or left >= right:
        return
    rows, cols = _ROWS[top:bottom, left:right], _COLUMNS[top:bottom, left:right]
    radii = np.hypot(cols - x, rows - y)
    pen = radii <= radius
    if inner_radius:
        pen &= radii >= inner_radius
    image[top:bottom, left:right][pen] = color


def _line(image, start, end, color, *, width):
    count = math.ceil(np.max(np.abs(end - start))) + 1
    positions = np.rint(np.linspace(start, end, count)).astype(int)
    for offset in width:
        points = positions + offset
        points = points[((0 <= points) & (points < RGB_CAMERA_SIZE)).all(axis=1)]
        image[points[:, 1], points[:, 0]] = color


_CROSS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
_GRID_WIDTH = ((0, 0),)


@lru_cache(maxsize=512)
def _floor(view, tick):
    matrix, origin = view.at(tick)
    image = _BACKGROUND.copy()
    for position in (-.72, -.48, -.24, 0., .24, .48, .72):
        _line(image, matrix @ (position, -.98) + origin, matrix @ (position, .98) + origin,
              (38, 46, 57), width=_GRID_WIDTH)
        _line(image, matrix @ (-.98, position) + origin, matrix @ (.98, position) + origin,
              (38, 46, 57), width=_GRID_WIDTH)
    return matrix, origin, image


def render_camera(view, tick, tip, markers, reference, segments=(), *, fiducials=()):
    """Rasterize the camera scene as one unsigned RGB image."""
    if len(fiducials) not in (0, len(FIDUCIAL_COLORS)):
        raise ValueError('supply either zero or three camera fiducials')
    matrix, origin, floor = _floor(view, tick)

    def project(point):
        return matrix @ point + origin

    image = floor.copy()
    for start, end in segments:
        _line(image, project(start), project(end), (143, 153, 160), width=_CROSS)
        _circle(image, project(start), 3., (171, 178, 184))

    brightness = int(3 * math.sin(tick * .59 + view.phase[1]))
    tint = np.add(view.illumination, brightness)
    _circle(image, project(reference), 2.2, np.add(REFERENCE_COLOR, tint))
    radius = PHYSICAL_RING_RADIUS * np.linalg.norm(matrix[0])
    for marker in markers:
        _circle(image, project(marker), radius, np.add(GOAL_COLOR, tint), inner_radius=radius - 1.6)
    for point, color in zip(fiducials, FIDUCIAL_COLORS):
        _circle(image, project(point), 2.4, np.add(color, tint))
    _circle(image, project(tip), 2.1, np.clip(np.add(TIP_COLOR, tint), 0, 255))
    return image


@dataclass(frozen=True)
class CameraObservation:
    tip: np.ndarray
    markers: np.ndarray
    reference: np.ndarray
    fiducials: np.ndarray


def _xy_pixels(mask):
    rows, cols = np.nonzero(mask)
    return np.column_stack((cols, rows))


def _two_centers(pixels):
    if len(pixels) < 10:
        raise ValueError('both goal rings must be visible')
    # The candidate spacing exceeds the largest ring width along at least one
    # image axis. Find the widest empty strip and measure each ring separately.
    coordinates = np.sort(pixels, axis=0)
    gaps = np.diff(coordinates, axis=0)
    index, axis = np.unravel_index(gaps.argmax(), gaps.shape)
    if gaps[index, axis] > 1:
        group = pixels[:, axis] <= coordinates[index, axis]
    else:
        # At lower optical zoom the empty strip can be narrower than one pixel.
        endpoint = pixels[np.square(pixels - pixels[0]).sum(axis=1).argmax()]
        distances = np.square(pixels - endpoint).sum(axis=1)
        other_endpoint = pixels[distances.argmax()]
        group = distances <= np.square(pixels - other_endpoint).sum(axis=1)
    first, second = pixels[group], pixels[~group]
    if min(len(first), len(second)) < 8:
        raise ValueError('camera did not resolve two separate goal rings')
    return np.stack((first.mean(axis=0), second.mean(axis=0)))


def decode_camera(frame):
    """Return painted marker centroids using color thresholds on RGB pixels."""
    if frame.shape != (RGB_CAMERA_SIZE, RGB_CAMERA_SIZE, 3) or frame.dtype != np.uint8:
        raise ValueError('expected an unsigned RGB camera image')
    r, g, b = np.moveaxis(frame, -1, 0)
    coordinates = _xy_pixels((r > 196) | (g > 190) | (b > 200))
    cols, rows = coordinates.T
    r, g, b = frame[rows, cols].T
    tip = coordinates[(r > 222) & (g > 156) & (b < 101)]
    reference = coordinates[(r < 120) & (g > 185) & (b > 201)]
    if len(tip) < 4 or len(reference) < 4:
        raise ValueError('camera lost a painted tip or reference marker')
    markers = _two_centers(coordinates[(r > 197) & (g < 122) & (b > 151) & (b < 219)])
    fiducial_masks = (
        (r < 113) & (g > 198) & (b > 81) & (b < 127),
        (r < 131) & (g > 85) & (g < 137) & (b > 222),
        (r > 200) & (g > 192) & (b > 147) & (b < 194),
    )
    fiducials = [coordinates[mask] for mask in fiducial_masks]
    if any(len(points) for points in fiducials):
        if any(len(points) < 4 for points in fiducials):
            raise ValueError('camera lost a painted fiducial')
        fiducials = np.stack([points.mean(axis=0) for points in fiducials])
    else:
        fiducials = np.empty((0, 2))
    return CameraObservation(tip.mean(axis=0), markers, reference.mean(axis=0), fiducials)


@dataclass(frozen=True)
class Drive:
    delay: int
    mixing: tuple
    response: tuple
    speed_limit: tuple

    @classmethod
    def sample(cls, rng, *, changed=False):
        if changed:
            order = rng.choice(((1, 2, 0), (2, 0, 1)))
            signs = rng.choice((-1., 1.), 3)
            mix = np.eye(3)[order] * signs[:, None] + rng.uniform(-.13, .13, (3, 3))
            return cls(int(rng.integers(1, MAX_COMMAND_DELAY + 1)), tuple(map(tuple, mix)),
                       tuple(rng.uniform(.42, .75, 3)), tuple(rng.uniform(1.0, 1.65, 3)))
        mix = np.eye(3) + rng.uniform(-.10, .10, (3, 3))
        return cls(int(rng.integers(0, 2)), tuple(map(tuple, mix)),
                   tuple(rng.uniform(.60, .72, 3)), tuple(rng.uniform(1.35, 1.85, 3)))


class CameraArena:
    """Evaluate contacts privately and return only camera frames and scalar reward."""

    def __init__(self, layout, paying, camera, *, max_steps, drive=None, wrist=0.):
        self.layout = layout
        self.paying = paying
        self.camera = camera
        self.max_steps = max_steps
        self.drive = drive
        self.wrist = wrist
        self._goals = layout.by_distance()
        self._mixing = None if drive is None else np.asarray(drive.mixing)
        self._joints = None if drive is None else _configuration_for(layout.start, wrist)
        self._tip = layout.start
        self._delay = deque([np.zeros(3)] * (0 if drive is None else drive.delay))
        self.trace = [self._tip]
        self.contacts = []
        self._dwell = [0, 0]
        self.success = False

    @property
    def steps(self):
        return len(self.trace) - 1

    @property
    def done(self):
        return self.success or self.steps >= self.max_steps

    def frame(self):
        if self.drive is None:
            x, y = self._tip
            segments = (((-.86, y), (.86, y)), ((x, -.78), (x, .78)))
        else:
            points = _joint_points(self._joints)
            segments = tuple(zip(points, points[1:]))
        return render_camera(self.camera, self.steps, self._tip, self.layout.markers,
                             self.layout.reference, segments)

    def step(self, command):
        if self.done:
            raise RuntimeError('reaching episode has ended')
        command = np.asarray(command, dtype=float)
        expected = 2 if self.drive is None else 3
        if command.shape != (expected,) or not np.isfinite(command).all() or np.max(np.abs(command)) > 1.:
            raise ValueError(f'expected {expected} finite normalized relative commands')
        if self.drive is None:
            self._tip = tuple(np.clip(self._tip + .043 * command, -.78, .78))
        else:
            self._delay.append(command)
            applied = self._mixing @ self._delay.popleft()
            setpoints = _relative_setpoint(self._joints, applied)
            joints = []
            for angle, target, response, speed in zip(self._joints, setpoints,
                                                       self.drive.response, self.drive.speed_limit):
                change = response * (target - angle)
                cap = speed * TICK_SECONDS
                joints.append(angle + min(cap, max(-cap, change)))
            self._joints = tuple(joints)
            self._tip = _joint_points(self._joints)[-1]
        self.trace.append(self._tip)
        for goal, marker in enumerate(self._goals):
            self._dwell[goal] = self._dwell[goal] + 1 if math.dist(self._tip, marker) <= GOAL_TOLERANCE else 0
            if self._dwell[goal] == SUCCESS_TICKS:
                self.contacts.append({'tick': self.steps, 'relation': goal, 'reward': int(goal == self.paying)})
                self.success |= goal == self.paying
        return self.frame(), int(self.success)
