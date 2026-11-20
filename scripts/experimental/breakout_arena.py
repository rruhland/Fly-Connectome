"""Native-scale Breakout with the Pong camera, motor commands, and cadence."""

from collections import deque
from dataclasses import dataclass, replace
import hashlib
import math
import random

import torch

from fly_connectome.control.environment import CameraTransition, PongCamera
from fly_connectome.pong import Body, Physics


SIZE = 64
ROWS, COLUMNS = 3, 8
SPLITS = ("practice", "heldout")
EPSILON = 1e-12


def brick_bounds(row, column):
    """Inner arena coordinates of a rendered brick, in left/right/top/bottom order."""
    return ((2 + 8 * column) / SIZE, (7 + 8 * column) / SIZE,
            (5 + 6 * row) / SIZE, (8 + 6 * row) / SIZE)


def _derived_seed(seed, namespace, index=0):
    raw = hashlib.sha256(f"breakout-v1/{namespace}/{seed}/{index}".encode()).digest()
    return int.from_bytes(raw[:8], "big")


@dataclass(frozen=True)
class BreakoutLayout:
    cells: tuple[tuple[int, int], ...]

    def __post_init__(self):
        cells = tuple(tuple(cell) for cell in self.cells)
        if not cells or len(set(cells)) != len(cells) or any(
                type(row) is not int or type(column) is not int
                or not 0 <= row < ROWS or not 0 <= column < COLUMNS
                for row, column in cells):
            raise ValueError("one or more unique cells in the three-by-eight brick field required")
        object.__setattr__(self, "cells", tuple(sorted(cells)))

    @classmethod
    def from_rows(cls, rows):
        rows = tuple(tuple(row) for row in rows)
        if len(rows) != ROWS or any(len(row) != COLUMNS for row in rows) or any(
                occupied not in (0, 1) for row in rows for occupied in row):
            raise ValueError("three rows of eight binary brick occupancies required")
        return cls(tuple((row, column) for row in range(ROWS) for column in range(COLUMNS)
                         if rows[row][column]))

    @classmethod
    def generate(cls, seed, *, split="heldout"):
        if split not in SPLITS:
            raise ValueError("unknown layout split")
        source = random.Random(_derived_seed(seed, f"layout/{split}"))
        # Equal row counts and disjoint occupancy parity across splits.
        while True:
            cells = tuple((row, column) for row in range(ROWS)
                          for column in source.sample(range(COLUMNS), 3))
            if sum(column for _, column in cells) % 2 == SPLITS.index(split):
                return cls(cells)


@dataclass(frozen=True)
class BreakoutServe:
    x: float
    y: float
    vx: float
    vy: float

    def __post_init__(self):
        radius = Physics().ball_radius
        if (not all(math.isfinite(value) for value in (self.x, self.y, self.vx, self.vy))
                or not radius <= self.x <= 1 - radius or not radius <= self.y < 1 or not self.vy):
            raise ValueError("finite serve position inside the walls and nonzero vertical speed required")

    @classmethod
    def generate(cls, seed, *, side=None):
        """Start under the bricks, descending to a reachable signed landing strip."""
        if side is not None and (type(side) is not int or side not in (-1, 0, 1)):
            raise ValueError("serve side must be -1, 0, or 1")
        config = Physics()
        source = random.Random(_derived_seed(seed, "serve"))
        sampled_side = source.choice((-1, 1))
        side = sampled_side if side is None else side
        y = source.uniform(.39, .47)
        angle = source.uniform(.12, .285) * side
        vx, vy = config.ball_speed * math.sin(angle), config.ball_speed * math.cos(angle)
        distance = (source.uniform(.18, .28) * side if side else source.uniform(-.075, .075))
        target = .5 + distance
        plane = 1 - config.paddle_x - config.ball_radius
        return cls(target - (plane - y) / vy * vx, y, vx, vy)


@dataclass(frozen=True)
class BreakoutEpisode:
    seed: int
    layout_seed: int
    serve_side: int
    layout_split: str

    def camera(self, **kwargs):
        return BreakoutCamera(self.seed, layout_seed=self.layout_seed, serve_side=self.serve_side,
                              layout_split=self.layout_split, **kwargs)


def breakout_schedule(seed, lives, *, split="heldout", centered=False):
    """Make a schedule independent of the policy or the lengths of earlier lives."""
    if type(lives) is not int or lives < 1 or split not in SPLITS:
        raise ValueError("positive number of lives and known layout split required")
    source = random.Random(_derived_seed(seed, f"directions/{split}"))
    sides = []
    while len(sides) < lives:
        left_right = [-1, 1] + ([0] if centered else [])
        source.shuffle(left_right)
        sides.extend(left_right)
    return tuple(BreakoutEpisode(_derived_seed(seed, f"life/{split}", index),
                                 _derived_seed(seed, f"board/{split}", index), side, split)
                 for index, side in enumerate(sides[:lives]))


def _brick_entry(ball, velocity, bounds, limit):
    """Swept point against a brick already expanded by the ball's half extent."""
    entry, exit = [], []
    for position, speed, low, high in zip(ball, velocity, bounds[::2], bounds[1::2]):
        if speed == 0:
            if not low <= position <= high:
                return None
            entry.append(-math.inf)
            exit.append(math.inf)
        else:
            first, last = (low - position) / speed, (high - position) / speed
            entry.append(min(first, last))
            exit.append(max(first, last))
    time = max(entry)
    if -EPSILON <= time <= min(limit + EPSILON, *exit):
        # An exact corner contact reverses both velocity components.
        return max(time, 0.), tuple(math.isclose(axis_time, time, abs_tol=EPSILON, rel_tol=0.)
                                   for axis_time in entry)
    return None


class BreakoutCamera:
    """Evaluators own this object. Controllers receive the rendered camera transitions."""

    actions = PongCamera.actions

    def __init__(self, seed=0, *, layout_seed=None, layout=None, serve=None,
                 max_episode_ticks=None, activation_seconds=None, delay_frames=0,
                 mapping=(0, 1, 2), distractors=False, serve_side=None, layout_split="heldout"):
        if max_episode_ticks is not None and (
                type(max_episode_ticks) is not int or max_episode_ticks < 1):
            raise ValueError("positive episode tick limit required")
        if type(delay_frames) is not int or delay_frames < 0:
            raise ValueError("delay must count nonnegative camera commands")
        mapping = tuple(mapping)
        if len(mapping) != 3 or any(type(action) is not int for action in mapping) or set(mapping) != {0, 1, 2}:
            raise ValueError("mapping must be a permutation of the three native actions")
        self.config = Physics()
        if activation_seconds is not None:
            if not math.isfinite(activation_seconds) or activation_seconds <= 0:
                raise ValueError("positive finite activation time required")
            self.config = replace(self.config, tau_activation=activation_seconds)
        self.seed = seed
        self.layout = BreakoutLayout.generate(seed if layout_seed is None else layout_seed,
                                              split=layout_split) if layout is None else layout
        self.serve = serve
        self.serve_side = serve_side
        self.max_episode_ticks = max_episode_ticks
        self.delay_frames = delay_frames
        self.mapping = mapping
        self.distractors = distractors
        self.body = Body(1, self.config)
        self.paddle_y = 1 - self.config.paddle_x
        self._motor = tuple(torch.tensor([drive]) for drive in self.actions)
        self._body_mask = torch.ones(1, dtype=torch.bool)
        self._episode = -1
        self._reset()

    def _reset(self):
        self._episode += 1
        self._life_seed = self.seed if not self._episode else _derived_seed(self.seed, "reset", self._episode)
        launch = self.serve if self.serve is not None else BreakoutServe.generate(self._life_seed, side=self.serve_side)
        self.ball = [launch.x, launch.y]
        self.velocity = [launch.vx, launch.vy]
        self.body.reset(self._body_mask)
        self.ticks = self.decisions = self.returns = 0
        self.contacts, self.brick_events = [], []
        self.needs_reset = self.life_lost = self.cleared = self.truncated = False
        self._passed_paddle = launch.y > self.paddle_y - self.config.ball_radius and launch.vy > 0
        self._queue = deque([1] * self.delay_frames)
        self._bricks = {}
        self._background = torch.zeros((SIZE, SIZE), dtype=torch.bool)
        radius = self.config.ball_radius
        for cell in self.layout.cells:
            x0, x1, y0, y1 = brick_bounds(*cell)
            self._bricks[cell] = (x0 - radius, x1 + radius, y0 - radius, y1 + radius)
            self._background[round(y0 * SIZE):round(y1 * SIZE), round(x0 * SIZE):round(x1 * SIZE)] = True
        if self.distractors:
            self._background[24:27, 2:5] = True
            self._background[40:43, 58:61] = True

    def reset(self):
        """Begin another independently seeded life on a restored board."""
        self._reset()
        return self.frame()

    @staticmethod
    def _pixels(position, half):
        start = math.ceil((position - half) * SIZE - .5)
        end = math.floor((position + half) * SIZE - .5) + 1
        return slice(max(0, min(SIZE, start)), max(0, min(SIZE, end)))

    def frame(self):
        frame = self._background.clone()
        frame[self._pixels(self.paddle_y, .0125),
              self._pixels(float(self.body.position[0]), self.config.paddle_half)] = True
        frame[self._pixels(self.ball[1], self.config.ball_radius),
              self._pixels(self.ball[0], self.config.ball_radius)] = True
        return frame

    def _remove_brick(self, cell, time):
        del self._bricks[cell]
        x0, x1, y0, y1 = brick_bounds(*cell)
        self._background[round(y0 * SIZE):round(y1 * SIZE), round(x0 * SIZE):round(x1 * SIZE)] = False
        self.brick_events.append(dict(tick=self.ticks, camera=self.decisions + 1, time_seconds=time,
                                      row=cell[0], column=cell[1], after_return=self.returns > 0))
        self.cleared = not self._bricks

    def _paddle_contact(self, time):
        position = float(self.body.position[0])
        offset = self.ball[0] - position
        success = abs(offset) <= self.config.paddle_half + self.config.ball_radius
        self.contacts.append(dict(tick=self.ticks, camera=self.decisions + 1, time_seconds=time,
                                  success=success, ball=self.ball.copy(), velocity=self.velocity.copy(),
                                  ball_x=self.ball[0], paddle_x=position, offset=offset))
        if success:
            self.returns += 1
            angle = .9 * max(-1., min(1., offset / self.config.paddle_half))
            self.velocity[:] = (self.config.ball_speed * math.sin(angle),
                                -self.config.ball_speed * math.cos(angle))
        else:
            self._passed_paddle = True

    def _collision(self, remaining):
        x, y = self.ball
        vx, vy = self.velocity
        radius = self.config.ball_radius
        candidates = []
        if vx:
            candidates.append(((radius - x if vx < 0 else 1 - radius - x) / vx, "side", None))
        if vy < 0:
            candidates.append(((radius - y) / vy, "ceiling", None))
        if vy > 0:
            if not self._passed_paddle:
                candidates.append(((self.paddle_y - radius - y) / vy, "paddle", None))
            candidates.append(((1 - y) / vy, "floor", None))
        if min(y, y + remaining * vy) <= brick_bounds(ROWS - 1, 0)[3] + radius:
            for cell, bounds in self._bricks.items():
                entry = _brick_entry(self.ball, self.velocity, bounds, remaining)
                if entry is not None:
                    candidates.append((entry[0], "brick", (cell, entry[1])))
        return min((event for event in candidates if -EPSILON <= event[0] <= remaining + EPSILON),
                   key=lambda event: event[0], default=None)

    def _advance_ball(self):
        remaining = self.config.dt
        while remaining >= 0:
            event = self._collision(remaining)
            duration = remaining if event is None else max(0., event[0])
            for axis in range(2):
                self.ball[axis] += duration * self.velocity[axis]
            if event is None:
                return
            remaining -= duration
            time = self.ticks * self.config.dt - remaining
            _, kind, payload = event
            if kind == "side":
                self.velocity[0] *= -1
            elif kind == "ceiling":
                self.velocity[1] *= -1
            elif kind == "paddle":
                self._paddle_contact(time)
            elif kind == "floor":
                self.life_lost = True
                return
            else:
                cell, axes = payload
                self._remove_brick(cell, time)
                for axis, reflect in enumerate(axes):
                    if reflect:
                        self.velocity[axis] *= -1
                if self.cleared:
                    return

    def step(self, action):
        if self.needs_reset:
            raise ValueError("reset required after termination or truncation")
        if type(action) is not int or not 0 <= action < len(self.actions):
            raise ValueError("legal actuator index required")
        self._queue.append(self.mapping[action])
        drive = self._motor[self._queue.popleft()]
        budget = (self.decisions + 1) * 12 // 5 - self.decisions * 12 // 5
        prior_returns, prior_bricks = self.returns, len(self.brick_events)
        for ticks in range(1, budget + 1):
            self.body.step(drive)
            self.ticks += 1
            self._advance_ball()
            terminated = self.life_lost or self.cleared
            self.truncated = bool(not terminated and self.max_episode_ticks is not None
                                  and self.ticks >= self.max_episode_ticks)
            if terminated or self.truncated:
                break
        self.decisions += 1
        self.needs_reset = terminated or self.truncated
        return CameraTransition(self.frame(), float(len(self.brick_events) - prior_bricks),
                                ticks * self.config.dt, terminated, self.truncated, ticks,
                                self.returns - prior_returns)

    def summary(self):
        """Evaluation-only outcomes, including primary and serve-earned score."""
        contacts = [dict(contact, ball=contact["ball"].copy(), velocity=contact["velocity"].copy())
                    for contact in self.contacts]
        goal = len(self.layout.cells)
        serve_bricks = sum(not event["after_return"] for event in self.brick_events)
        return dict(seed=self.seed, life_seed=self._life_seed, life_index=self._episode,
                    cameras=self.decisions, ticks=self.ticks,
                    elapsed=self.ticks * self.config.dt, first_incoming=contacts[0] if contacts else None,
                    contacts=contacts, brick_events=[dict(event) for event in self.brick_events],
                    own_opportunities=len(contacts), own_returns=self.returns, life_lost=self.life_lost,
                    terminated=self.life_lost or self.cleared, cleared=self.cleared, truncated=self.truncated,
                    unfinished_incoming_approach=bool(self.truncated and self.velocity[1] > 0
                                                     and not self._passed_paddle),
                    initial_bricks=goal, remaining_bricks=len(self._bricks), score=len(self.brick_events),
                    serve_bricks=serve_bricks, post_own_bricks=len(self.brick_events) - serve_bricks)


def play_breakout(controller, seed=0, *, max_episode_ticks=2400, **kwargs):
    """Play one complete life through a raw-camera controller or a passive paddle."""
    camera = BreakoutCamera(seed, max_episode_ticks=max_episode_ticks, **kwargs)
    if controller is not None:
        controller.reset(camera.frame())
    neutral = camera.mapping.index(1)
    requests = bytearray()
    while not camera.needs_reset:
        action = neutral if controller is None else controller.choose()[0]
        transition = camera.step(action)
        requests.append(action)
        if controller is not None:
            controller.observe(transition)
    return dict(camera.summary(), action_sha256=hashlib.sha256(requests).hexdigest())
