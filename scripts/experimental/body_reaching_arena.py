"""Matched reaching tasks with binary visual markers and private simulator state."""

from dataclasses import dataclass, replace
import math

import numpy as np


CAMERA_SIZE = 97
CAMERA_CENTER = (CAMERA_SIZE - 1) / 2
TICK_SECONDS = .04
GOAL_TOLERANCE = .055
SUCCESS_TICKS = 4
DEFAULT_MAX_STEPS = 96
TASK_X = (.22, .63)
TASK_Y = (-.36, .36)
MIN_START_DISTANCE = .26


@dataclass(frozen=True)
class ReachCase:
    """Start and target in a common reachable rectangle, in camera world units."""

    start: tuple[float, float]
    target: tuple[float, float]

    def __post_init__(self):
        for name in ('start', 'target'):
            point = tuple(float(value) for value in getattr(self, name))
            if (len(point) != 2 or not TASK_X[0] <= point[0] <= TASK_X[1]
                    or not TASK_Y[0] <= point[1] <= TASK_Y[1]):
                raise ValueError(f'{name} must be inside the common reaching workspace')
            object.__setattr__(self, name, point)
        if self.start == self.target:
            raise ValueError('start and target must differ')


def sample_case(rng):
    """Draw a matched case from the shared workspace with a nontrivial displacement."""
    while True:
        start = (rng.uniform(*TASK_X), rng.uniform(*TASK_Y))
        target = (rng.uniform(*TASK_X), rng.uniform(*TASK_Y))
        if math.dist(start, target) >= MIN_START_DISTANCE:
            return ReachCase(start, target)


@dataclass(frozen=True)
class BodySpec:
    """The drive maps motor commands to linear or angular setpoint velocities."""

    limits: tuple[tuple[float, float], tuple[float, float]]
    drive: tuple[tuple[float, float], tuple[float, float]]
    response: tuple[float, float]
    links: tuple[float, float] | None = None


_CARRIAGE = BodySpec(limits=((-.73, .83), (-.78, .78)),
                     drive=((1.15, 0.), (0., 1.05)), response=(.80, .79))
_ARM = BodySpec(limits=((-2.62, 1.24), (.20, 2.90)),
                drive=((1.65, 0.), (0., 2.05)), response=(.78, .73), links=(.46, .42))
_ARM_VARIANT = replace(_ARM, drive=((1.18, .30), (-.39, 1.48)), response=(.55, .61))

BODY_SPECS = {
    'carriage': _CARRIAGE,
    'arm': _ARM,
    'carriage_variant': replace(_CARRIAGE, drive=((.72, -.24), (.19, .87)), response=(.58, .69)),
    'arm_variant': _ARM_VARIANT,
    # Map external (a, b) to the variant's motor inputs (-b, a).
    'arm_remapped': replace(_ARM_VARIANT, drive=((.30, -1.18), (1.48, .39))),
}
BODY_NAMES = tuple(BODY_SPECS)
DEFAULT_BODY_NAMES = ('carriage', 'arm', 'arm_variant')


def camera_pixel(position):
    """Project a camera-world position to integer (row, column) in the raster."""
    x, y = position
    return round((1 - y) * CAMERA_CENTER), round((1 + x) * CAMERA_CENTER)


def render_markers(position, target):
    """Rasterize a square tip and a hollow square goal on separate binary planes.

    The marker planes keep both visible when the tip overlaps the goal. The
    sensor contains no pixels for the body's joints, links, rails, or actuators.
    The physical workspace leaves enough margin for both markers at its edges.
    """
    frame = np.zeros((2, CAMERA_SIZE, CAMERA_SIZE), dtype=bool)
    tip_row, tip_col = camera_pixel(position)
    goal_row, goal_col = camera_pixel(target)
    frame[0, tip_row - 1:tip_row + 2, tip_col - 1:tip_col + 2] = True
    frame[1, goal_row - 2:goal_row + 3, goal_col - 2:goal_col + 3] = True
    frame[1, goal_row - 1:goal_row + 2, goal_col - 1:goal_col + 2] = False
    return frame


def decode_markers(frame):
    """Decode the marker centers from binary pixels, in x-right and y-up units.

    Each plane contains a single known-size marker whose first lit pixel is its
    upper left corner. The exact center therefore needs only two pixel reads.
    """
    if frame.shape != (2, CAMERA_SIZE, CAMERA_SIZE):
        raise ValueError('expected two binary marker planes at camera resolution')
    planes = frame.reshape(2, -1)
    tip_pixel, goal_pixel = np.argmax(planes, axis=1)
    if not (planes[0, tip_pixel] and planes[1, goal_pixel]):
        raise ValueError('both visual markers must be present')
    tip_row, tip_col = divmod(int(tip_pixel), CAMERA_SIZE)
    goal_row, goal_col = divmod(int(goal_pixel), CAMERA_SIZE)
    position = ((tip_col + 1) / CAMERA_CENTER - 1, 1 - (tip_row + 1) / CAMERA_CENTER)
    target = ((goal_col + 2) / CAMERA_CENTER - 1, 1 - (goal_row + 2) / CAMERA_CENTER)
    return np.array(position, dtype=np.float32), np.array(target, dtype=np.float32)


def _arm_configuration(position, links):
    """Reset the arm on the positive elbow branch of its planar kinematics."""
    x, y = position
    upper, lower = links
    elbow = math.acos((x * x + y * y - upper * upper - lower * lower) / (2 * upper * lower))
    shoulder = math.atan2(y, x) - math.atan2(lower * math.sin(elbow), upper + lower * math.cos(elbow))
    return shoulder, elbow


def _arm_points(configuration, links):
    shoulder, elbow = configuration
    upper, lower = links
    hinge = (upper * math.cos(shoulder), upper * math.sin(shoulder))
    tip = (hinge[0] + lower * math.cos(shoulder + elbow),
           hinge[1] + lower * math.sin(shoulder + elbow))
    return hinge, tip


class ReachingArena:
    """One episode. Only camera-derived observations belong in the controller.

    Evaluator-owned fields expose the true tip path and geometry scores for
    auditing. Four consecutive ticks inside the true tolerance end an episode
    successfully. A time limit ends any episode that has not met that criterion.
    """

    def __init__(self, body, case, *, max_steps=DEFAULT_MAX_STEPS):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError('max_steps must be a positive integer')
        self.body = body
        self.case = case
        self.max_steps = max_steps
        self._spec = BODY_SPECS[body]
        if self._spec.links is None:
            self._configuration = case.start
        else:
            self._configuration = _arm_configuration(case.start, self._spec.links)
        self._velocity = (0., 0.)
        self._trace = [self._tip_position()]
        self._initial_error = math.dist(self._trace[0], case.target)
        self._error = self._initial_error
        self._weighted_error_sum = 0.
        self._inside_ticks = 0

    @property
    def steps(self):
        return len(self._trace) - 1

    @property
    def success(self):
        return self._inside_ticks >= SUCCESS_TICKS

    @property
    def done(self):
        return self.success or self.steps >= self.max_steps

    @property
    def trace(self):
        return tuple(self._trace)

    @property
    def ground_truth_position(self):
        return self._trace[-1]

    @property
    def error(self):
        return self._error

    @property
    def normalized_error(self):
        return self.error / self._initial_error

    def _weight_sum(self, steps):
        return steps + steps * (steps - 1) / (2 * max(1, self.max_steps - 1))

    @property
    def weighted_normalized_error(self):
        """Mean start-normalized true error with step weights increasing from 1 to 2.

        Remaining ticks use the last observed error, so early success has the
        same time horizon as timeout. A stationary miss has score 1.
        """
        total_weight = self._weight_sum(self.max_steps)
        remaining_weight = total_weight - self._weight_sum(self.steps)
        return (self._weighted_error_sum + remaining_weight * self.normalized_error) / total_weight

    def _tip_position(self):
        if self._spec.links is None:
            return self._configuration
        return _arm_points(self._configuration, self._spec.links)[1]

    def frame(self):
        return render_markers(self.ground_truth_position, self.case.target)

    def observation(self):
        return decode_markers(self.frame())

    def step(self, action):
        """Advance the actuators by one 40 ms tick and return a fresh visual observation."""
        if self.done:
            raise RuntimeError('the reaching episode has ended')
        action_x, action_y = tuple(float(value) for value in action)
        if not math.isfinite(action_x) or not math.isfinite(action_y):
            raise ValueError('motor commands must be finite')
        action_x = min(1., max(-1., action_x))
        action_y = min(1., max(-1., action_y))
        positions, velocities = [], []
        for position, velocity, limits, drive, response in zip(
                self._configuration, self._velocity, self._spec.limits,
                self._spec.drive, self._spec.response):
            demand = drive[0] * action_x + drive[1] * action_y
            velocity += response * (demand - velocity)
            position += TICK_SECONDS * velocity
            if not limits[0] <= position <= limits[1]:
                position = min(limits[1], max(limits[0], position))
                velocity = 0.
            positions.append(position)
            velocities.append(velocity)
        self._configuration = tuple(positions)
        self._velocity = tuple(velocities)
        tip = self._tip_position()
        self._trace.append(tip)
        self._error = math.dist(tip, self.case.target)
        self._inside_ticks = self._inside_ticks + 1 if self.error <= GOAL_TOLERANCE else 0
        weight = 1 + (self.steps - 1) / max(1, self.max_steps - 1)
        self._weighted_error_sum += weight * self.normalized_error
        return self.observation()

    def render_rgb(self, *, scale=4):
        """Render evaluator-only body geometry, a true tip trail, and camera markers."""
        if type(scale) is not int or scale < 1:
            raise ValueError('scale must be a positive integer')
        image = np.empty((CAMERA_SIZE, CAMERA_SIZE, 3), dtype=np.uint8)
        image[:] = (15, 22, 33)
        image[::12, :, :] = (25, 36, 53)
        image[:, ::12, :] = (25, 36, 53)
        tip = self.ground_truth_position
        link_color = (150, 170, 197)
        if self._spec.links is None:
            x_limits, y_limits = self._spec.limits
            for x in x_limits:
                _draw_segment(image, (x, y_limits[0]), (x, y_limits[1]), link_color)
            for y in y_limits:
                _draw_segment(image, (x_limits[0], y), (x_limits[1], y), link_color)
            _draw_segment(image, (x_limits[0], tip[1]), (x_limits[1], tip[1]), link_color)
            _draw_segment(image, (tip[0], y_limits[0]), (tip[0], y_limits[1]), link_color)
        else:
            hinge, _ = _arm_points(self._configuration, self._spec.links)
            _draw_segment(image, (0., 0.), hinge, link_color, thickness=1)
            _draw_segment(image, hinge, tip, link_color, thickness=1)
            for joint in ((0., 0.), hinge):
                row, col = camera_pixel(joint)
                image[row - 2:row + 3, col - 2:col + 3] = (232, 239, 242)
        for start, end in zip(self._trace, self._trace[1:]):
            _draw_segment(image, start, end, (67, 129, 169))
        tip_marker, goal_marker = self.frame()
        image[goal_marker] = (95, 225, 170)
        image[tip_marker] = (255, 191, 116)
        return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


def _draw_segment(image, start, end, color, *, thickness=0):
    row_start, col_start = camera_pixel(start)
    row_end, col_end = camera_pixel(end)
    count = max(abs(row_end - row_start), abs(col_end - col_start)) + 1
    rows = np.rint(np.linspace(row_start, row_end, count)).astype(int)
    cols = np.rint(np.linspace(col_start, col_end, count)).astype(int)
    for row_offset in range(-thickness, thickness + 1):
        for col_offset in range(-thickness, thickness + 1):
            image[rows + row_offset, cols + col_offset] = color
