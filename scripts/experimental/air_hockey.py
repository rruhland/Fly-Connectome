"""Two-dimensional air hockey with a rendered, reward-only agent interface."""

from collections import deque
from dataclasses import dataclass
import math
import random

import numpy as np


WIDTH, HEIGHT = 1.0, .75
PIXELS_PER_UNIT = 128
HZ = 60
PUCK_RADIUS = .021
MALLET_RADIUS = .047
GOAL_HALF = .117
ACTIONS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
           (-1, -1), (1, -1), (-1, 1), (1, 1))
PLAYER_COLOR = (240, 103, 95)
OPPONENT_COLOR = (104, 167, 252)
PUCK_COLOR = (255, 237, 168)
GOAL_COLOR = (93, 216, 189)
RAIL_COLOR = (132, 159, 171)
SUBSTEPS = 4
MAX_PUCK_SPEED = 1.85


@dataclass(frozen=True)
class HockeyServe:
    x: float
    y: float
    vx: float
    vy: float

    @classmethod
    def generate(cls, rng, restitution, style):
        # Independently generated counterattack starts. Mixed starts alternate
        # in distribution between direct and banked threats to the own goal.
        if style == 'mixed':
            style = rng.choice(('bank', 'direct'))
        if style == 'neutral':
            return cls(.5, rng.uniform(.23, .52), 0., 0.)
        vx = -rng.uniform(1.22, 1.51) if style == 'fast-bank' else -rng.uniform(.58, .84)
        x = rng.uniform(.58, .65) if style == 'fast-bank' else rng.uniform(.76, .86)
        sign = rng.choice((-1, 1))
        y_goal = HEIGHT / 2 + sign * rng.uniform(.068, .091)
        if style == 'direct':
            y = rng.uniform(.12, HEIGHT - .12)
            return cls(x, y, vx, (y_goal - y) / ((x - PUCK_RADIUS) / -vx))
        if style not in ('bank', 'fast-bank'):
            raise ValueError('unknown serve style')
        boundary = PUCK_RADIUS if sign < 0 else HEIGHT - PUCK_RADIUS
        y = rng.uniform(.13, .62) if style == 'fast-bank' else rng.uniform(.25, .61)
        first_leg = abs(boundary - y)
        second_leg = abs(boundary - y_goal)
        vy = sign * (first_leg + second_leg / max(restitution, .05)) / ((x - PUCK_RADIUS) / -vx)
        scale = MAX_PUCK_SPEED / max(MAX_PUCK_SPEED, math.hypot(vx, vy))
        return cls(x, y, vx * scale, vy * scale)


def _draw_disc(frame, x, y, radius, color):
    cx, cy = x * PIXELS_PER_UNIT, y * PIXELS_PER_UNIT
    r = radius * PIXELS_PER_UNIT
    left = max(0, math.ceil(cx - r - .5))
    top = max(0, math.ceil(cy - r - .5))
    right = min(frame.shape[1], math.floor(cx + r - .5) + 1)
    bottom = min(frame.shape[0], math.floor(cy + r - .5) + 1)
    xx = (np.arange(left, right) + .5 - cx) ** 2
    yy = (np.arange(top, bottom) + .5 - cy) ** 2
    frame[top:bottom, left:right][yy[:, None] + xx <= r * r] = color


def _table_image():
    frame = np.full((round(HEIGHT * PIXELS_PER_UNIT), round(WIDTH * PIXELS_PER_UNIT), 3),
                    (17, 32, 46), dtype=np.uint8)
    middle = frame.shape[1] // 2
    frame[::3, middle - 1:middle + 1] = (41, 76, 96)
    _draw_disc(frame, WIDTH / 2, HEIGHT / 2, .12, (41, 76, 96))
    _draw_disc(frame, WIDTH / 2, HEIGHT / 2, .110, (17, 32, 46))
    slot = slice(round((HEIGHT / 2 - GOAL_HALF) * PIXELS_PER_UNIT),
                 round((HEIGHT / 2 + GOAL_HALF) * PIXELS_PER_UNIT))
    frame[[0, -1]] = RAIL_COLOR
    frame[:, [0, 1, -2, -1]] = RAIL_COLOR
    frame[slot, 0:2] = GOAL_COLOR
    frame[slot, -2:] = GOAL_COLOR
    return frame


TABLE = _table_image()


class AirHockeyCamera:
    """The player observes RGB pixels, its requested button, and public goal reward."""

    actions = ACTIONS

    def __init__(self, seed=0, *, max_frames=540, opponent='guard', wall_restitution=1.,
                 serve_style='mixed', serve=None):
        if type(max_frames) is not int or max_frames < 1:
            raise ValueError('positive frame budget required')
        if opponent not in ('guard', 'still', 'press', 'laneguard'):
            raise ValueError('unknown opponent')
        if serve_style not in ('direct', 'bank', 'fast-bank', 'mixed', 'neutral'):
            raise ValueError('unknown serve style')
        if not math.isfinite(wall_restitution) or not .05 <= wall_restitution <= 1.:
            raise ValueError('wall restitution must be between .05 and 1')
        self.seed = seed
        self.max_frames = max_frames
        self.wall_restitution = wall_restitution
        self.serve_style = serve_style
        self.opponent = opponent
        self.serve = serve
        self._round = -1
        self.reset()

    def reset(self):
        self._round += 1
        rng = random.Random(f'hockey/v1/{self.seed}/{self._round}')
        start = self.serve if self.serve is not None else HockeyServe.generate(
            rng, self.wall_restitution, self.serve_style)
        if (not all(math.isfinite(value) for value in (start.x, start.y, start.vx, start.vy))
                or not PUCK_RADIUS <= start.x <= WIDTH - PUCK_RADIUS
                or not PUCK_RADIUS <= start.y <= HEIGHT - PUCK_RADIUS
                or math.hypot(start.vx, start.vy) > MAX_PUCK_SPEED + 1e-12):
            raise ValueError('invalid serve')
        self.puck = [start.x, start.y]
        self.velocity = [start.vx, start.vy]
        self.player = [.245, HEIGHT / 2]
        self.rival = [.865, HEIGHT / 2 + rng.uniform(-.018, .018)]
        self._aim_side = rng.choice((-1, 1))
        if self.opponent == 'laneguard':
            self.rival = [.891, HEIGHT / 2 + .065 * self._aim_side]
        self.frames = self.reward = 0
        self.terminated = self.truncated = False
        self.events = []
        self._rival_memory = deque([(start.x, start.y, start.vx, start.vy)] * 12, maxlen=12)
        return self.frame()

    def frame(self):
        frame = TABLE.copy()
        _draw_disc(frame, *self.player, MALLET_RADIUS, PLAYER_COLOR)
        _draw_disc(frame, *self.rival, MALLET_RADIUS, OPPONENT_COLOR)
        _draw_disc(frame, *self.puck, PUCK_RADIUS, PUCK_COLOR)
        return frame

    @staticmethod
    def _body_step(position, command, minimum, maximum, speed, dt):
        vx, vy = command
        norm = max(1., math.hypot(vx, vy))
        old_x, old_y = position
        position[0] = min(maximum, max(minimum, old_x + dt * speed * vx / norm))
        position[1] = min(HEIGHT - MALLET_RADIUS,
                          max(MALLET_RADIUS, old_y + dt * speed * vy / norm))
        return (position[0] - old_x) / dt, (position[1] - old_y) / dt

    def _rival_command(self):
        if self.opponent == 'still':
            return (0., 0.), 0.
        x, y, vx, vy = self._rival_memory[0]
        if self.opponent == 'laneguard':
            lane = HEIGHT / 2 + .065 * self._aim_side
            forward = .891 - .11 * max(0., x - .5) if vx > 0. else .891
            target = lane + min(.034, max(-.034, .22 * (y - lane))) if vx > 0. else lane
            return ((forward - self.rival[0]) * 15, (target - self.rival[1]) * 15), .61
        pressing = self.opponent == 'press'
        home = .824 if pressing else .872
        if x >= .48 and abs(vx) < .12 and abs(vy) < .25:
            forward = min(home, max(.55, x + .04))
        elif x > .67 and vx > -.16:
            forward = min(home, max(.573, x + .065))
        else:
            forward = home
        if vx > .10:
            lead = min(.26, max(0., (forward - x) / vx))
            y += vy * lead
        if x >= .5:
            target = min(.59, max(.16, y + .028 * self._aim_side))
        else:
            target = HEIGHT / 2 + .025 * self._aim_side
        return ((forward - self.rival[0]) * 14, (target - self.rival[1]) * 14), .82 if pressing else .64

    def _impact(self, body, body_velocity, side, substep):
        dx, dy = self.puck[0] - body[0], self.puck[1] - body[1]
        distance = math.hypot(dx, dy)
        reach = PUCK_RADIUS + MALLET_RADIUS
        if distance >= reach:
            return
        if distance > 1e-10:
            nx, ny = dx / distance, dy / distance
        else:
            nx, ny = (1., 0.) if side == 'player' else (-1., 0.)
        self.puck[0] = body[0] + nx * (reach + 1e-9)
        self.puck[1] = body[1] + ny * (reach + 1e-9)
        normal_speed = ((self.velocity[0] - body_velocity[0]) * nx
                        + (self.velocity[1] - body_velocity[1]) * ny)
        if normal_speed >= 0.:
            return
        incoming = tuple(self.velocity)
        self.velocity[0] -= 1.95 * normal_speed * nx
        self.velocity[1] -= 1.95 * normal_speed * ny
        factor = MAX_PUCK_SPEED / max(MAX_PUCK_SPEED, math.hypot(*self.velocity))
        self.velocity[0] *= factor
        self.velocity[1] *= factor
        self._event('contact', substep, side=side, incoming=incoming)

    def _event(self, kind, substep, **data):
        self.events.append(dict(kind=kind, frame=self.frames + 1, substep=substep,
                                puck=tuple(self.puck), velocity=tuple(self.velocity), **data))

    def _rails(self, substep):
        x, y = self.puck
        if y < PUCK_RADIUS or y > HEIGHT - PUCK_RADIUS:
            boundary = PUCK_RADIUS if y < PUCK_RADIUS else HEIGHT - PUCK_RADIUS
            self.puck[1] = boundary - (y - boundary) * self.wall_restitution
            if (self.velocity[1] < 0.) == (y < PUCK_RADIUS):
                self.velocity[1] *= -self.wall_restitution
                self._event('rail', substep)
        if PUCK_RADIUS <= x <= WIDTH - PUCK_RADIUS:
            return
        scoring = x > WIDTH - PUCK_RADIUS
        if abs(self.puck[1] - HEIGHT / 2) < GOAL_HALF - PUCK_RADIUS:
            self.reward = 1 if scoring else -1
            self.terminated = True
            self._event('goal', substep, reward=self.reward)
            return
        boundary = WIDTH - PUCK_RADIUS if scoring else PUCK_RADIUS
        self.puck[0] = boundary - (x - boundary) * .92
        if (self.velocity[0] > 0.) == scoring:
            self.velocity[0] *= -.92
            self._event('end_rail', substep)

    def step(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError('reset after a goal or a time limit')
        if not isinstance(action, (int, np.integer)) or not 0 <= action < len(ACTIONS):
            raise ValueError('unknown action')
        rival_command, rival_speed = self._rival_command()
        dt = 1 / (HZ * SUBSTEPS)
        for substep in range(SUBSTEPS):
            player_velocity = self._body_step(self.player, ACTIONS[action], MALLET_RADIUS,
                                              WIDTH / 2 - MALLET_RADIUS, .98, dt)
            rival_velocity = self._body_step(self.rival, rival_command, WIDTH / 2 + MALLET_RADIUS,
                                             WIDTH - MALLET_RADIUS, rival_speed, dt)
            self.puck[0] += self.velocity[0] * dt
            self.puck[1] += self.velocity[1] * dt
            self._impact(self.player, player_velocity, 'player', substep)
            self._impact(self.rival, rival_velocity, 'rival', substep)
            self._rails(substep)
            if self.terminated:
                break
        self.frames += 1
        self._rival_memory.append((*self.puck, *self.velocity))
        self.truncated = self.frames == self.max_frames and not self.terminated
        return self.frame(), self.reward, self.terminated, self.truncated
