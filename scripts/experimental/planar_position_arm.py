"""Three-joint planar simulation with a bounded relative-position motor interface.

This generic simulation uses three rigid links and idealized position servos.
The camera sees the two reaching-task markers. Joint configuration, encoder
readings and requested positions belong to the simulated motor interface.
"""

import math

import numpy as np

from body_reaching_arena import (CAMERA_SIZE, DEFAULT_MAX_STEPS, GOAL_TOLERANCE, SUCCESS_TICKS,
                                 TICK_SECONDS, _draw_segment, camera_pixel, decode_markers,
                                 render_markers)


ACTION_DIM = 3
LINK_LENGTHS = (.40, .29, .20)
JOINT_LIMITS = ((-2.68, 1.28), (.15, 3.05), (-2.1, 2.1))
MAX_RELATIVE_SETPOINT = (.11, .13, .16)  # Radians per command.
SERVO_RESPONSE = (.60, .62, .64)  # Fraction of residual setpoint error per tick.
MAX_JOINT_SPEED = (1.48, 1.63, 1.93)  # Radians per second.
RESET_WRIST_LIMIT = .72


def _configuration_for(position, wrist):
    """Solve the shoulder and elbow with one chosen distal joint angle."""
    x, y = position
    upper, lower, tip = LINK_LENGTHS
    distal_x = lower + tip * math.cos(wrist)
    distal_y = tip * math.sin(wrist)
    distal_length = math.hypot(distal_x, distal_y)
    distal_offset = math.atan2(distal_y, distal_x)
    cosine = (x * x + y * y - upper * upper - distal_length * distal_length) / (2 * upper * distal_length)
    bend = math.acos(cosine)
    shoulder = math.atan2(y, x) - math.atan2(distal_length * math.sin(bend), upper + distal_length * cosine)
    return shoulder, bend - distal_offset, wrist


def _joint_points(configuration):
    points = [(0., 0.)]
    angle = 0.
    for relative_angle, length in zip(configuration, LINK_LENGTHS):
        angle += relative_angle
        x, y = points[-1]
        points.append((x + length * math.cos(angle), y + length * math.sin(angle)))
    return points


def _relative_setpoint(encoder, action):
    """Translate bounded commands using the simulated encoder sample at this tick."""
    return tuple(min(upper, max(lower, angle + scale * min(1., max(-1., command))))
                 for angle, command, scale, (lower, upper) in
                 zip(encoder, action, MAX_RELATIVE_SETPOINT, JOINT_LIMITS))


class Arena:
    """Reach at the same cadence and true hold threshold as the two-axis task.

    A seeded wrist angle changes the hidden reset posture at the same visible
    starting point. Each three-component action sets a new relative joint
    reference. Each servo closes a fraction of the remaining position error
    subject to its per-tick angular travel limit.
    """

    def __init__(self, case, *, posture_seed=0, max_steps=DEFAULT_MAX_STEPS):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError('max_steps must be a positive integer')
        self.case = case
        self.max_steps = max_steps
        wrist = np.random.default_rng(posture_seed).uniform(-RESET_WRIST_LIMIT, RESET_WRIST_LIMIT)
        self._configuration = _configuration_for(case.start, wrist)
        self._setpoint = self._configuration
        self._trace = [_joint_points(self._configuration)[-1]]
        self._initial_error = math.dist(self.ground_truth_position, self.case.target)
        self._error = self._initial_error
        self._inside_ticks = 0
        self._weighted_error_sum = 0.

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
    def commanded_setpoint(self):
        """Evaluator audit of the last actuator reference in radians."""
        return self._setpoint

    @property
    def error(self):
        return self._error

    @property
    def normalized_error(self):
        return self._error / self._initial_error

    def _weight_sum(self, steps):
        return steps + steps * (steps - 1) / (2 * max(1, self.max_steps - 1))

    @property
    def weighted_normalized_error(self):
        """Use the same full-horizon, start-normalized geometry score as ReachingArena."""
        total_weight = self._weight_sum(self.max_steps)
        remaining_weight = total_weight - self._weight_sum(self.steps)
        return (self._weighted_error_sum + remaining_weight * self.normalized_error) / total_weight

    def frame(self):
        return render_markers(self.ground_truth_position, self.case.target)

    def observation(self):
        return decode_markers(self.frame())

    def step(self, action):
        if self.done:
            raise RuntimeError('the reaching episode has ended')
        command = tuple(float(value) for value in action)
        if len(command) != ACTION_DIM or not all(math.isfinite(value) for value in command):
            raise ValueError('action must contain three finite motor commands')
        self._setpoint = _relative_setpoint(self._configuration, command)
        next_angles = []
        for angle, reference, response, max_speed in zip(
                self._configuration, self._setpoint, SERVO_RESPONSE, MAX_JOINT_SPEED):
            maximum = max_speed * TICK_SECONDS
            travel = response * (reference - angle)
            next_angles.append(angle + min(maximum, max(-maximum, travel)))
        self._configuration = tuple(next_angles)
        tip = _joint_points(self._configuration)[-1]
        self._trace.append(tip)
        self._error = math.dist(tip, self.case.target)
        self._inside_ticks = self._inside_ticks + 1 if self.error <= GOAL_TOLERANCE else 0
        weight = 1 + (self.steps - 1) / max(1, self.max_steps - 1)
        self._weighted_error_sum += weight * self.normalized_error
        return self.observation()

    def render_rgb(self, *, scale=4):
        """Render the three links and true trail for evaluator-side playback."""
        if type(scale) is not int or scale < 1:
            raise ValueError('scale must be a positive integer')
        image = np.empty((CAMERA_SIZE, CAMERA_SIZE, 3), dtype=np.uint8)
        image[:] = (15, 22, 33)
        image[::12, :, :] = (25, 36, 53)
        image[:, ::12, :] = (25, 36, 53)
        points = _joint_points(self._configuration)
        for start, end in zip(points, points[1:]):
            _draw_segment(image, start, end, (150, 170, 197), thickness=1)
            row, col = camera_pixel(start)
            image[row - 2:row + 3, col - 2:col + 3] = (232, 239, 242)
        for start, end in zip(self._trace, self._trace[1:]):
            _draw_segment(image, start, end, (67, 129, 169))
        tip_marker, goal_marker = self.frame()
        image[goal_marker] = (95, 225, 170)
        image[tip_marker] = (255, 191, 116)
        return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
