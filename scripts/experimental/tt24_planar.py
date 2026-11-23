"""A small continuous planar paddle with a fixed silhouette camera."""

from dataclasses import asdict, dataclass
import math

import numpy as np


DT = 1 / 60
STAGES = ('lateral', 'energy', 'damped')


@dataclass(frozen=True)
class Drop:
    x: float
    y: float
    vx: float
    vy: float
    gravity: float
    restitution: float
    half_span: float
    contrast: int


class PlanarBounce:
    """Integrate gravity and moving-paddle impacts, returning pixels and scalar score."""

    dt = DT
    radius = .028
    paddle_half_height = .025
    short_rise = .055
    width = 112
    height = 100
    _scale = 52.
    _baseline = .23
    _span = .16
    _x_limit = .88

    def __init__(self, stage='lateral', *, max_frames=240, presentation='fixed'):
        if stage not in STAGES or presentation not in ('fixed', 'varied'):
            raise ValueError('unknown planar stage or camera presentation')
        if max_frames < 1:
            raise ValueError('max_frames must be positive')
        self.stage = stage
        self.presentation = presentation
        self.action_size = 1 if stage == 'lateral' else 2
        self.max_frames = max_frames
        self._background = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def reset(self, seed):
        rng = np.random.default_rng(seed)
        vx = rng.uniform(.23, .63) * rng.choice((-1., 1.))
        elasticity = {'lateral': (.78, .87), 'energy': (.52, .65), 'damped': (.12, .20)}
        restitution = rng.uniform(*elasticity[self.stage])
        self._drop = Drop(float(rng.uniform(-.61, .61)), float(rng.uniform(.78, 1.31)), float(vx),
                          float(rng.uniform(-.48, .21)), float(rng.uniform(2.35, 3.2)), float(restitution),
                          float(rng.uniform(.9, 1.12) * self._span), int(rng.integers(83, 235)))
        self._seed = int(seed)
        self._x, self._y, self._vx, self._vy = self._drop.x, self._drop.y, self._drop.vx, self._drop.vy
        self._px, self._py = 0., self._baseline
        self._flight = None
        self._contacts = []
        self._rebounds = []
        self._count = 0
        self._t = 0.
        self._done = False
        self._reason = None
        self._transform = (False, 0, 0, 0)
        if self.presentation == 'varied':
            self._transform = (bool(rng.integers(2)), int(rng.integers(5, 39)),
                               int(rng.integers(-4, 5)), int(rng.integers(-4, 5)))
        self._background.fill(self._transform[1])
        return self.render()

    def step(self, action):
        if self._done:
            raise RuntimeError('reset before advancing a finished drop')
        drive = np.asarray(action, dtype=float)
        if drive.shape != (self.action_size,) or not np.isfinite(drive).all():
            raise ValueError('expected one finite rate for each free actuator')
        dx = max(-1., min(1., float(drive[0]))) * (1.55 * self.dt)
        dy = max(-1., min(1., float(drive[1]))) * (1.05 * self.dt) if self.action_size == 2 else 0.
        next_x = max(-self._x_limit, min(self._x_limit, self._px + dx))
        next_y = max(.10, min(.69, self._py + dy))
        paddle_vx, paddle_vy = (next_x - self._px) / self.dt, (next_y - self._py) / self.dt
        gravity = self._drop.gravity
        floor = self.radius + self.paddle_half_height
        gap = self._y - self._py - floor
        gap_end = gap + (self._vy - paddle_vy) * self.dt - .5 * gravity * self.dt ** 2
        reward = 0.

        if gap >= -1.e-10 and gap_end <= 0.:
            approach = self._vy - paddle_vy
            impact = (approach + math.sqrt(max(0., approach * approach + 2. * gravity * gap))) / gravity
            impact = max(0., min(self.dt, impact))
            hit_x = self._x + impact * self._vx
            center_x = self._px + impact * paddle_vx
            if abs(hit_x - center_x) <= self._drop.half_span + .42 * self.radius:
                reward += self._flight_reward(impact)
                # Relative normal restitution and limited tangential impulse.
                incoming = self._vy - gravity * impact
                outgoing = paddle_vy + self._drop.restitution * (paddle_vy - incoming)
                hit_y = self._py + impact * paddle_vy + floor
                self._x, self._y = hit_x, hit_y
                self._vx += max(-.23, min(.23, .13 * (paddle_vx - self._vx)))
                self._vy = outgoing
                self._flight = None
                if incoming - paddle_vy < -.20 and outgoing > .20:
                    self._flight = (self._t + impact, hit_y, incoming, outgoing, paddle_vy)
                self._contacts.append(self._t + impact)
                remaining = self.dt - impact
                reward += self._flight_reward(remaining, delay=impact)
                self._flight_step(remaining)
            else:
                reward += self._flight_reward(self.dt)
                self._flight_step(self.dt)
        else:
            reward += self._flight_reward(self.dt)
            self._flight_step(self.dt)

        self._px, self._py = next_x, next_y
        self._count += 1
        self._t += self.dt
        # Fixed Pong walls reflect only the tangential component.
        boundary = 1. - self.radius
        if abs(self._x) > boundary:
            self._x = math.copysign(2. * boundary - abs(self._x), self._x)
            self._vx *= -.91
        if self._y < self.radius:
            self._reason = 'fell'
        elif self._y > 1.76:
            self._reason = 'ceiling_exit'
        elif self._count >= self.max_frames:
            self._reason = 'limit'
        self._done = self._reason is not None
        return self.render(), reward, self._done

    def _flight_step(self, duration):
        g = self._drop.gravity
        self._x += self._vx * duration
        self._y += self._vy * duration - .5 * g * duration ** 2
        self._vy -= g * duration

    def _flight_reward(self, duration, *, delay=0.):
        if self._flight is None or not 0 < self._vy / self._drop.gravity <= duration:
            return 0.
        impact, height, incoming, outgoing, blade_rate = self._flight
        time = self._vy / self._drop.gravity
        apex = self._y + .5 * self._vy * time
        self._flight = None
        rise = apex - height
        if rise < self.short_rise:
            return 0.
        self._rebounds.append(dict(impact_time=impact, apex_time=self._t + delay + time, rise=rise,
                                   incoming=incoming, outgoing=outgoing, paddle_rate=blade_rate))
        return 1.

    def _pixel(self, x, y):
        mirror, _, tx, ty = self._transform
        return (round((self.width - 1) / 2 + (-x if mirror else x) * self._scale + tx),
                round(self.height - 5 - y * self._scale + ty))

    def render(self):
        frame = self._background.copy()
        body_x, body_y = self._pixel(self._px, self._py)
        span = round(self._drop.half_span * self._scale)
        thickness = round(self.paddle_half_height * self._scale)
        low_x, high_x = max(0, body_x - span), min(self.width, body_x + span + 1)
        low_y, high_y = max(0, body_y - thickness), min(self.height, body_y + thickness + 1)
        if low_x < high_x and low_y < high_y:
            frame[low_y:high_y, low_x:high_x] = self._drop.contrast
        x, y = self._pixel(self._x, self._y)
        radius = max(2, round(self.radius * self._scale))
        for dy in range(-radius, radius + 1):
            if 0 <= y + dy < self.height:
                half = math.isqrt(radius ** 2 - dy ** 2)
                start, stop = max(0, x - half), min(self.width, x + half + 1)
                if start < stop:
                    frame[y + dy, start:stop] = self._drop.contrast
        return frame

    def judge(self):
        return dict(seed=self._seed, presentation=self.presentation, stage=self.stage, drop=asdict(self._drop),
                    frames=self._count, rebounds=len(self._rebounds), contacts=list(self._contacts),
                    events=list(self._rebounds), termination=self._reason)
