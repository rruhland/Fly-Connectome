"""Three joint servos and two spring-return switches with axial travel."""

from dataclasses import dataclass
import math

import numpy as np

from arm_plays_pong import NATIVE_HZ
from planar_position_arm import LINK_LENGTHS, _relative_setpoint
from pong_button_panel import (
    BUTTON_COLORS, PRESS_RADIUS, SOURCE_GEAR, TIP_COLOR, ButtonPanel, _PositionArm, decode_panel,
)


BOARD_Z = 0.
HOME_Z = .009
MAX_TIP_Z = .024
PROFILE_SCALE = 1200  # Vertical pixels per metre in the virtual side camera.
PROFILE_ORIGIN = 41
PROFILE_MARKER_OFFSET = .007
PROFILE_HEIGHT = 54
PROFILE_WIDTH = 160


def _points(joints):
    shoulder, elbow, pitch = joints
    first, second, wrist = LINK_LENGTHS
    p1 = (first * math.cos(shoulder), first * math.sin(shoulder))
    heading = shoulder + elbow
    c, s = math.cos(heading), math.sin(heading)
    p2 = (p1[0] + second * c, p1[1] + second * s)
    planar_wrist = wrist * math.cos(pitch)
    return ((0., 0.), p1, p2, (p2[0] + planar_wrist * c, p2[1] + planar_wrist * s))


def _pitch_for_height(height):
    return math.asin(height / LINK_LENGTHS[-1])


class PitchArm(_PositionArm):
    """Two planar links and a hinged wrist. Wrist pitch changes tip z and xy."""

    def __init__(self, start):
        super().__init__(start, 0., tick_seconds=1/NATIVE_HZ, gear=SOURCE_GEAR)
        self._joints = (*self._joints[:2], _pitch_for_height(HOME_Z))
        self.points = _points(self._joints)

    @property
    def height(self):
        return LINK_LENGTHS[-1] * math.sin(self._joints[2])

    def step(self, command, switches=()):
        action = np.asarray(command, dtype=float)
        if action.shape != (3,) or not np.isfinite(action).all() or np.max(np.abs(action)) > 1.:
            raise ValueError('expected three finite relative joint commands in [-1, 1]')
        setpoint = _relative_setpoint(self._joints, action)
        # This end-effector operates inside a limited angular envelope around the board.
        setpoint = (*setpoint[:2], min(_pitch_for_height(MAX_TIP_Z), max(_pitch_for_height(-.008), setpoint[2])))
        old = self._joints
        proposed = tuple(angle + min(cap, max(-cap, response * (reference - angle)))
                         for angle, reference, response, cap in zip(old, setpoint, self.response, self.travel_limits))
        previous_cap = _cap_under(self.tip, switches)
        intended_cap = _cap_under(_points(proposed)[-1], switches)
        if self.height < BOARD_Z and previous_cap is not None and intended_cap != previous_cap:
            # A fingertip below the panel face has to retract through the lip.
            proposed = (*old[:2], proposed[2])
            intended_cap = previous_cap
        lowest = BOARD_Z if intended_cap is None else -switches[intended_cap].travel
        if switches:
            proposed = (*proposed[:2], max(proposed[2], _pitch_for_height(lowest)))
        self._joints = proposed
        self.points = _points(proposed)


@dataclass(frozen=True)
class Switch:
    center: tuple[float, float]
    travel: float
    closes_at: float
    opens_at: float
    return_speed: float


def _cap_under(tip, switches):
    return next((i for i, switch in enumerate(switches) if math.dist(tip, switch.center) <= PRESS_RADIUS), None)


def profile_frame(tip, height, neutral, switches=(), depths=()):
    """Rasterize a magnified, orthographic side view of the cap and fingertip marks."""
    image = np.zeros((PROFILE_HEIGHT, PROFILE_WIDTH, 3), dtype=np.uint8)
    image[:] = (25, 31, 37)
    image[PROFILE_ORIGIN, :] = (106, 116, 126)

    def mark(y, z, color, half_width):
        col = round(PROFILE_WIDTH / 2 + 310 * (y - neutral[1]))
        row = round(PROFILE_ORIGIN - PROFILE_SCALE * z)
        left, right = max(0, col - half_width), min(PROFILE_WIDTH, col + half_width + 1)
        top, bottom = max(0, row - 1), min(PROFILE_HEIGHT, row + 2)
        if left < right and top < bottom:
            image[top:bottom, left:right] = color

    for switch, depth, color in zip(switches, depths, BUTTON_COLORS, strict=True):
        mark(switch.center[1], -depth, color, 6)
    mark(tip[1], height + PROFILE_MARKER_OFFSET, TIP_COLOR, 2)
    return image


@dataclass(frozen=True)
class PanelFrame:
    overhead: np.ndarray
    profile: np.ndarray


@dataclass(frozen=True)
class TravelObservation:
    tip: np.ndarray
    buttons: np.ndarray
    neutral: np.ndarray
    height: float
    depth: tuple[float, float]


def decode_travel(frame):
    panel = decode_panel(frame.overhead)
    rows = []
    for color in (TIP_COLOR, *BUTTON_COLORS):
        y, _ = np.nonzero(np.all(frame.profile == color, axis=2))
        if not len(y):
            raise ValueError('side camera lost a painted depth marker')
        rows.append(float(y.mean()))
    return TravelObservation(panel.tip, panel.buttons, panel.neutral,
                             (PROFILE_ORIGIN - rows[0]) / PROFILE_SCALE - PROFILE_MARKER_OFFSET,
                             tuple((row - PROFILE_ORIGIN) / PROFILE_SCALE for row in rows[1:]))


class TravelButtonPanel(ButtonPanel):
    """Use physical cap depth for both closing and reopening each switch."""

    def __init__(self, seed, *, wiring_seed):
        super().__init__(seed, wiring_seed=wiring_seed, gear=SOURCE_GEAR, tick_seconds=1/NATIVE_HZ)
        self._arm = PitchArm(self.neutral)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 26_861]))
        self.switches = tuple(Switch(center, travel, travel * rng.uniform(.60, .72),
                                     travel * rng.uniform(.24, .42), rng.uniform(.11, .17))
                              for center, travel in zip(self.buttons, rng.uniform(.0036, .0052, 2).tolist(), strict=True))
        self.depth = [0., 0.]
        self.closed = [False, False]
        self.overlap = None
        self.max_depth = [0., 0.]
        self.shallow_overlap_ticks = 0
        self.depth_only_releases = 0
        self.transitions = []

    @property
    def height(self):
        return self._arm.height

    def frame(self):
        return PanelFrame(self._camera.frame(self._arm),
                          profile_frame(self.tip, self.height, self.neutral, self.switches, self.depth))

    def step(self, command):
        self._arm.step(command, self.switches)
        self.steps += 1
        self.overlap = _cap_under(self.tip, self.switches)
        height = self.height
        for i, switch in enumerate(self.switches):
            contact = min(switch.travel, max(0., -height)) if i == self.overlap else 0.
            depth = max(contact, self.depth[i] - switch.return_speed / NATIVE_HZ)
            self.depth[i] = depth
            self.max_depth[i] = max(depth, self.max_depth[i])
            was_closed = self.closed[i]
            threshold = switch.opens_at if was_closed else switch.closes_at
            is_closed = bool(depth > threshold if was_closed else depth >= threshold)
            self.closed[i] = is_closed
            if i == self.overlap and not is_closed:
                self.shallow_overlap_ticks += 1
            if was_closed != is_closed:
                self.transitions.append(dict(tick=self.steps, button=i, closed=is_closed,
                                             depth_m=depth, tip_z_m=height, overlap=i == self.overlap))
                if was_closed:
                    self.depth_only_releases += int(i == self.overlap)
        self.active_button = self.closed.index(True) if sum(self.closed) == 1 else None
        return self.output
