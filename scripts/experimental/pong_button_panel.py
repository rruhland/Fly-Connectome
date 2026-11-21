"""A two-button physical controller for Pong with RGB-only visual servoing."""

from dataclasses import dataclass
import math

import numpy as np

from body_reaching_arena import TICK_SECONDS as REACHING_TICK_SECONDS
from camera_motor_learner import MotorModel
from camera_reaching_arena import (
    CameraView,
    OPTICAL_SCALE_RANGE,
    RGB_CAMERA_SIZE,
    TIP_COLOR,
    _circle,
    _floor,
    _line,
)
from planar_position_arm import (
    ACTION_DIM,
    MAX_JOINT_SPEED,
    RESET_WRIST_LIMIT,
    SERVO_RESPONSE,
    _configuration_for,
    _joint_points,
    _relative_setpoint,
)


PONG_TICK_SECONDS = 1 / 120
SOURCE_GEAR = 1.
FAST_GEAR = REACHING_TICK_SECONDS / PONG_TICK_SECONDS
PRESS_RADIUS = .039
RELEASE_RADIUS = .049
PRESS_TICKS = 2
_NOMINAL_CAMERA_SCALE = sum(OPTICAL_SCALE_RANGE) / len(OPTICAL_SCALE_RANGE)
VISUAL_BUTTON_RADIUS = PRESS_RADIUS * _NOMINAL_CAMERA_SCALE
VISUAL_RELEASE_RADIUS = RELEASE_RADIUS * _NOMINAL_CAMERA_SCALE

BUTTON_COLORS = ((235, 83, 183), (69, 227, 99))
NEUTRAL_COLOR = (96, 221, 237)
_LINK_WIDTH = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
_PROBE_STRENGTH = .65
_PROBE_TICKS = 2 * ACTION_DIM
_MODEL_DISCOUNT = .91
_PIXEL_DEADBAND = .75


class _PositionArm:
    """The reaching arm's relative servos, run at the panel's physical cadence."""

    def __init__(self, start, wrist, *, tick_seconds, gear):
        if not (math.isfinite(tick_seconds) and tick_seconds > 0 and math.isfinite(gear) and gear > 0):
            raise ValueError('arm tick duration and gearing must be finite and positive')
        effective_dt = tick_seconds * gear
        self.travel_limits = tuple(speed * effective_dt for speed in MAX_JOINT_SPEED)
        self.response = tuple(1. - (1. - response) ** (effective_dt / REACHING_TICK_SECONDS)
                              for response in SERVO_RESPONSE)
        self._joints = _configuration_for(start, wrist)
        self.points = _joint_points(self._joints)

    @property
    def tip(self):
        return self.points[-1]

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all() or np.max(np.abs(action)) > 1.:
            raise ValueError('expected three finite relative joint commands in [-1, 1]')
        setpoints = _relative_setpoint(self._joints, action)
        angles = []
        for angle, reference, response, cap in zip(self._joints, setpoints, self.response, self.travel_limits):
            travel = response * (reference - angle)
            angles.append(angle + min(cap, max(-cap, travel)))
        self._joints = tuple(angles)
        self.points = _joint_points(self._joints)


class _PanelCamera:
    def __init__(self, view, buttons, neutral):
        self._matrix, self._origin, self._floor = _floor(view, 0)
        self._tint = view.illumination
        self._buttons = tuple(self._project(point) for point in buttons)
        self._neutral = self._project(neutral)
        self._radius = RELEASE_RADIUS * view.scale + .75

    def _project(self, point):
        return self._matrix @ point + self._origin

    def _color(self, color):
        return tuple(min(255, channel + tint) for channel, tint in zip(color, self._tint))

    def frame(self, arm):
        image = self._floor.copy()
        points = tuple(self._project(point) for point in arm.points)
        for start, end in zip(points, points[1:]):
            _line(image, start, end, (143, 153, 160), width=_LINK_WIDTH)
            _circle(image, start, 2.8, (171, 178, 184))
        for point, color in zip(self._buttons, BUTTON_COLORS):
            _circle(image, point, self._radius, (48, 53, 65))
            _circle(image, point, self._radius, self._color(color), inner_radius=self._radius - 1.4)
        _circle(image, self._neutral, 3.3, self._color(NEUTRAL_COLOR), inner_radius=2.1)
        _circle(image, points[-1], 2.2, self._color(TIP_COLOR))
        return image


class ButtonPanel:
    """Latch a mapped paddle input only after consecutive physical contacts.

    The fingertip and switch centers share world coordinates in the simulator.
    The switch opens once the fingertip leaves a wider release radius. Wiring
    is seeded independently of the seen scene, with positive output meaning
    screen down. Only the harness reads `step`, `active_button`, or events.
    The motor controller receives `frame()`.
    """

    def __init__(self, seed=0, *, wiring_seed=None, tick_seconds=PONG_TICK_SECONDS, gear=FAST_GEAR):
        scene_seed, independent_wiring_seed = np.random.SeedSequence(seed).spawn(2)
        rng = np.random.default_rng(scene_seed)
        wire_rng = np.random.default_rng(independent_wiring_seed if wiring_seed is None else wiring_seed)
        self.wiring = tuple(map(int, wire_rng.permutation((-1, 1))))
        self.tick_seconds = tick_seconds
        self.gear = gear
        self.neutral = (.585 + rng.uniform(-.020, .020), rng.uniform(-.025, .025))
        angle = rng.uniform(-.18, .18)
        spacing = rng.uniform(.073, .080)
        offset = spacing * np.array((math.sin(angle), math.cos(angle)))
        self.buttons = tuple(tuple(self.neutral + direction * offset) for direction in (-1., 1.))
        wrist = rng.uniform(-RESET_WRIST_LIMIT, RESET_WRIST_LIMIT)
        self._arm = _PositionArm(self.neutral, wrist, tick_seconds=tick_seconds, gear=gear)
        self._camera = _PanelCamera(CameraView.sample(rng), self.buttons, self.neutral)
        self.active_button = None
        self._candidate = None
        self._dwell = 0
        self.steps = 0
        self.press_events = []
        self.release_events = []
        self.wiring_events = []

    @property
    def tip(self):
        """Evaluator view of the fingertip in physical coordinates."""
        return self._arm.tip

    @property
    def output(self):
        return 0 if self.active_button is None else self.wiring[self.active_button]

    def frame(self):
        return self._camera.frame(self._arm)

    def rewire(self, wiring_seed=None):
        """Change the hidden wiring at the current physical instant."""
        if wiring_seed is None:
            self.wiring = self.wiring[::-1]
        else:
            self.wiring = tuple(map(int, np.random.default_rng(wiring_seed).permutation((-1, 1))))
        self.wiring_events.append({'tick': self.steps, 'wiring': self.wiring})

    def step(self, command):
        self._arm.step(command)
        self.steps += 1
        distances = tuple(math.dist(self.tip, center) for center in self.buttons)
        if self.active_button is not None:
            if distances[self.active_button] <= RELEASE_RADIUS:
                return self.output
            self.release_events.append({'tick': self.steps, 'button': self.active_button, 'output': self.output})
            self.active_button = None
        candidate = next((index for index, distance in enumerate(distances) if distance <= PRESS_RADIUS), None)
        if candidate is None:
            self._candidate = None
            self._dwell = 0
            return 0
        self._dwell = self._dwell + 1 if self._candidate == candidate else 1
        self._candidate = candidate
        if self._dwell >= PRESS_TICKS:
            self.active_button = candidate
            self.press_events.append({'tick': self.steps, 'button': candidate, 'output': self.output})
        return self.output


@dataclass(frozen=True)
class PanelObservation:
    tip: np.ndarray
    buttons: np.ndarray
    neutral: np.ndarray


def decode_panel(frame):
    """Read the painted fingertip, button identities, and neutral from RGB pixels."""
    if frame.shape != (RGB_CAMERA_SIZE, RGB_CAMERA_SIZE, 3) or frame.dtype != np.uint8:
        raise ValueError('expected an unsigned RGB arm-camera frame')
    red, green, blue = np.moveaxis(frame, -1, 0)
    rows, columns = np.nonzero((red > 200) | (green > 200))
    red, green, blue = frame[rows, columns].T
    coordinates = np.column_stack((columns, rows))
    masks = (
        (red > 240) & (green > 171) & (blue < 95),
        (red > 215) & (green < 105) & (blue > 160),
        (red < 90) & (green > 208) & (blue < 115),
        (red < 118) & (green > 200) & (blue > 215),
    )
    highlights = [coordinates[mask] for mask in masks]
    if any(len(points) < 4 for points in highlights):
        raise ValueError('arm camera lost a painted marker')
    tip, first, second, neutral = (points.mean(axis=0) for points in highlights)
    return PanelObservation(tip, np.stack((first, second)), neutral)


class VisualButtonMotor:
    """Fit local image motion from issued actions using discounted least squares.

    A cold controller probes each joint in both directions once. Every valid
    consecutive image pair then contributes its observed fingertip motion to
    a three-input, two-output linear model. Images provide the steering error.
    Desired targets are painted identities or the painted neutral location.
    """

    def __init__(self, seed=0, *, memory=None):
        self.model = MotorModel.empty(ACTION_DIM, 0)
        self.samples = 0
        if memory is not None:
            gram, cross = np.asarray(memory['gram'], dtype=float), np.asarray(memory['cross'], dtype=float)
            if (gram.shape != (ACTION_DIM, ACTION_DIM) or cross.shape != (ACTION_DIM, 2)
                    or not np.isfinite(gram).all() or not np.isfinite(cross).all()
                    or not np.allclose(gram, gram.T) or np.linalg.eigvalsh(gram).min() < -1e-10):
                raise ValueError('invalid arm motor memory')
            self.model.gram = gram.copy()
            self.model.cross = cross.copy()
            self.samples = int(memory['samples'])
            if self.samples < _PROBE_TICKS:
                raise ValueError('arm motor memory must include a full calibration')
        rng = np.random.default_rng(seed)
        axes = np.eye(ACTION_DIM)[rng.permutation(ACTION_DIM)]
        first = _PROBE_STRENGTH * axes * rng.choice((-1., 1.), size=(ACTION_DIM, 1))
        self._probes = np.stack((first, -first), axis=1).reshape(_PROBE_TICKS, ACTION_DIM)
        self._probe_index = 0
        self._previous_tip = None
        self._previous_command = None
        self._tick = 0
        self.probe_ticks = []

    def export_state(self):
        """Return only the visual-motor sufficient statistics for reuse."""
        return {'gram': self.model.gram.tolist(), 'cross': self.model.cross.tolist(), 'samples': self.samples}

    def _learn(self, tip):
        if self._previous_tip is not None and np.any(self._previous_command):
            self.model.observe(self._previous_command, tip - self._previous_tip, discount=_MODEL_DISCOUNT)
            self.samples += 1

    def act(self, frame, target, *, observation=None):
        if target is not None and (isinstance(target, bool) or not isinstance(target, (int, np.integer))
                                   or target not in (0, 1)):
            raise ValueError('arm target must be visual button 0, visual button 1, or neutral (None)')
        self._tick += 1
        if observation is None:
            try:
                observation = decode_panel(frame)
            except ValueError:
                self._previous_tip = None
                self._previous_command = None
                return np.zeros(ACTION_DIM)
        tip = observation.tip - observation.neutral
        self._learn(tip)
        if self.samples < _PROBE_TICKS:
            command = self._probes[self._probe_index % _PROBE_TICKS].copy()
            self._probe_index += 1
            self.probe_ticks.append(self._tick)
        else:
            goal = observation.neutral if target is None else observation.buttons[target]
            error = goal - observation.tip
            command = self._steer(error)
        self._previous_tip = tip
        self._previous_command = command.copy()
        return command

    def _steer(self, error):
        if np.linalg.norm(error) <= _PIXEL_DEADBAND:
            return np.zeros(ACTION_DIM)
        effects = self.model.effects()
        normal = effects @ effects.T + .18 * np.eye(2)
        command = effects.T @ np.linalg.solve(normal, .86 * error)
        return command / max(1., np.max(np.abs(command)))


def pretrain_motor(seed, *, ticks=80, tick_seconds=PONG_TICK_SECONDS, gear=FAST_GEAR):
    """Spend a fixed budget reaching painted marks in a separate arm-only scene.

    Practice has three scattered targets and no button electronics or game.
    The motor observes RGB and its own commands under the same cadence and
    gearing as the test arm. Its returned memory contains no task associations.
    """
    if type(ticks) is not int or ticks < _PROBE_TICKS + 1:
        raise ValueError('practice must allow at least one full calibration')
    rng = np.random.default_rng(seed)
    start = (.585 + rng.uniform(-.020, .020), rng.uniform(-.025, .025))
    standard = np.array(((.64, -.15), (.53, .09), (.64, .11)))
    marks = standard + rng.uniform(-.018, .018, standard.shape)
    scene = _PanelCamera(CameraView.sample(rng), marks[:2], marks[2])
    wrist = rng.uniform(-RESET_WRIST_LIMIT, RESET_WRIST_LIMIT)
    arm = _PositionArm(start, wrist, tick_seconds=tick_seconds, gear=gear)
    motor = VisualButtonMotor(seed)
    target_index = 0
    reached = 0
    for _ in range(ticks):
        target = (0, 1, None)[target_index]
        command = motor.act(scene.frame(arm), target)
        arm.step(command)
        if math.dist(arm.tip, marks[target_index]) <= .025:
            target_index = (target_index + 1) % len(marks)
            reached += 1
    return motor.export_state(), {'ticks': ticks, 'targets_reached': reached, 'probes': len(motor.probe_ticks)}
