"""A three-dimensional, camera-observed table tennis arena with joint-rate control."""

from dataclasses import dataclass
from functools import lru_cache
import math
import random

import numpy as np


HZ = 60
SUBSTEPS = 4
GRAVITY = 9.81
TABLE_HALF_WIDTH = .7625
TABLE_HALF_LENGTH = 1.37
NET_HEIGHT = .1525
BALL_RADIUS = .027
PADDLE_RADIUS = .143
TABLE_RESTITUTION = .86
PADDLE_RESTITUTION = .99
INITIAL_JOINTS = (0., -.50, 1.65)
JOINT_LIMITS = ((-1.18, 1.18), (-1.15, .75), (.52, 2.38))
JOINT_RATES = (3.5, 3.4, 4.5)
BASE = (0., -1.86, .11)
UPPER_ARM, FOREARM = .52, .50
IMAGE_HEIGHT, IMAGE_WIDTH = 112, 192
PIXELS_PER_METER = 44.
ORIGIN_TOP = (IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2)
ORIGIN_SIDE = (IMAGE_WIDTH / 2, 79.)
VARIANTS = ('standard', 'shifted', 'height')


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _unit(v):
    norm = math.sqrt(_dot(v, v))
    return tuple(value / norm for value in v)


def _clamp(value, bounds):
    return min(bounds[1], max(bounds[0], value))


def arm_pose(joints):
    """Return shoulder, elbow, paddle center and paddle-front normal in world coordinates."""
    yaw, shoulder, elbow = joints
    sin_yaw, cos_yaw = math.sin(yaw), math.cos(yaw)
    upper_depth = UPPER_ARM * math.cos(shoulder)
    lower_depth = FOREARM * math.cos(shoulder + elbow)
    elbow_point = (BASE[0] + upper_depth * sin_yaw, BASE[1] + upper_depth * cos_yaw,
                   BASE[2] + UPPER_ARM * math.sin(shoulder))
    paddle = (elbow_point[0] + lower_depth * sin_yaw, elbow_point[1] + lower_depth * cos_yaw,
              elbow_point[2] + FOREARM * math.sin(shoulder + elbow))
    # A wrist linkage keeps the blade facing the opposite court as the arm flexes.
    normal = _unit((-.20 * sin_yaw, 1., .315 + .075 * (shoulder + .50)))
    return BASE, elbow_point, paddle, normal


@dataclass(frozen=True)
class TT3Case:
    seed: int
    view_seed: int
    variant: str = 'standard'
    distractor: bool = False


def heldout_cases(seed, count, *, variant='standard', distractor=False):
    """Materialize a reproducible schedule with separate physics and appearance seeds."""
    if variant not in VARIANTS or type(count) is not int or count < 1:
        raise ValueError('expected a known variant and a positive case count')
    rng = random.Random(f'tt3/heldout/v1/{seed}/{variant}')
    return tuple(TT3Case(rng.getrandbits(63), rng.getrandbits(63), variant, distractor) for _ in range(count))


def _opening(rng, variant):
    """Generate a legal approaching ball from a reach envelope and a tabletop bounce."""
    shifted = variant == 'shifted'
    height_band = rng.choice(('low', 'high')) if variant == 'height' else None
    width = (.21, .62) if shifted else (.15, .55)
    depth_range = (-1.51, -1.10) if shifted else (-1.43, -1.13)
    heights = (.16, .49) if shifted else (.21, .43)
    speed = (3.55, 4.7) if shifted else (3.35, 4.25)
    rise_time = (.12, .22)
    launch_depth = (.45, .87)
    launch_height = (.15, .72)
    if height_band == 'low':
        depth_range, heights, rise_time = (-1.54, -1.16), (.11, .18), (.055, .12)
        launch_depth, launch_height = (.31, .85), (.09, .82)
    if height_band == 'high':
        depth_range, heights, rise_time = (-1.51, -1.14), (.53, .66), (.18, .29)
        launch_depth, launch_height = (.68, 1.05), (.15, 1.21)
    while True:
        hit_x = rng.choice((-1., 1.)) * rng.uniform(*width)
        hit_y, hit_z = rng.uniform(*depth_range), rng.uniform(*heights)
        depth = hit_y - BASE[1]
        yaw = math.atan2(hit_x, depth)
        radial = math.hypot(hit_x, depth)
        height = hit_z - BASE[2]
        cos_elbow = (radial**2 + height**2 - UPPER_ARM**2 - FOREARM**2) / (2 * UPPER_ARM * FOREARM)
        if not -1. < cos_elbow < 1.:
            continue
        elbow = math.acos(cos_elbow)
        shoulder = math.atan2(height, radial) - math.atan2(FOREARM * math.sin(elbow),
                                                          UPPER_ARM + FOREARM * cos_elbow)
        if any(not low <= joint <= high for joint, (low, high) in
               zip((yaw, shoulder, elbow), JOINT_LIMITS)):
            continue
        vy = -rng.uniform(*speed)
        travel_after_bounce = rng.uniform(*rise_time)
        bounce_y = hit_y - vy * travel_after_bounce
        if not -1.30 < bounce_y < -.41:
            continue
        start_y = rng.uniform(*launch_depth)
        start_x = .42 * hit_x + rng.uniform(-.32, .32)
        travel_before_bounce = (bounce_y - start_y) / vy
        vx = (hit_x - start_x) / (travel_before_bounce + travel_after_bounce)
        upward = (hit_z - BALL_RADIUS + GRAVITY * travel_after_bounce**2 / 2) / travel_after_bounce
        before_bounce = -upward / TABLE_RESTITUTION
        start_vz = before_bounce + GRAVITY * travel_before_bounce
        start_z = BALL_RADIUS - before_bounce * travel_before_bounce - GRAVITY * travel_before_bounce**2 / 2
        to_net = -start_y / vy
        net_z = start_z + start_vz * to_net - GRAVITY * to_net**2 / 2
        peak = start_z + max(0., start_vz)**2 / (2 * GRAVITY)
        if launch_height[0] < start_z < launch_height[1] and peak < 1.23 and net_z > NET_HEIGHT + BALL_RADIUS + .04:
            return (start_x, start_y, start_z), (vx, vy, start_vz)


def _project(point, camera):
    x, y, z = point
    if camera == 0:
        return ORIGIN_TOP[0] + y * PIXELS_PER_METER, ORIGIN_TOP[1] + x * PIXELS_PER_METER
    return ORIGIN_SIDE[0] + y * PIXELS_PER_METER, ORIGIN_SIDE[1] - z * PIXELS_PER_METER


def _ellipse(image, center, radius, color):
    x, y = center
    rx, ry = radius
    left, right = max(0, math.ceil(x - rx)), min(IMAGE_WIDTH, math.floor(x + rx) + 1)
    top, bottom = max(0, math.ceil(y - ry)), min(IMAGE_HEIGHT, math.floor(y + ry) + 1)
    if left >= right or top >= bottom:
        return
    xx = ((np.arange(left, right) - x) / rx)**2
    yy = ((np.arange(top, bottom) - y) / ry)**2
    image[top:bottom, left:right][yy[:, None] + xx <= 1.] = color


def _line(image, start, end, color, width=1.):
    steps = 1 + math.ceil(max(abs(a - b) for a, b in zip(start, end)))
    x, y = np.rint(np.linspace(start, end, steps)).astype(int).T
    x = np.clip(x, 1, IMAGE_WIDTH - 2)
    y = np.clip(y, 1, IMAGE_HEIGHT - 2)
    image[y, x] = color
    if width > 1.:
        image[y - 1, x] = image[y + 1, x] = image[y, x - 1] = image[y, x + 1] = color


@lru_cache(maxsize=2)
def _background(camera):
    image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), (23, 29, 37), dtype=np.uint8)
    edge, table, net = (110, 126, 129), (49, 75, 78), (97, 114, 123)
    a, b = TABLE_HALF_WIDTH, TABLE_HALF_LENGTH

    def line(start, end, color, width=1.):
        _line(image, _project(start, camera), _project(end, camera), color, width)

    if camera == 0:
        (left, top), (right, bottom) = _project((-a, -b, 0), camera), _project((a, b, 0), camera)
        image[round(top):round(bottom) + 1, round(left):round(right) + 1] = table
        for x in (-a, a):
            line((x, -b, 0), (x, b, 0), edge)
        for y in (-b, b):
            line((-a, y, 0), (a, y, 0), edge)
        line((0, -b, 0), (0, b, 0), (68, 92, 93))
        line((-a - .04, 0, 0), (a + .04, 0, 0), net, 2.)
    else:
        image[101:] = (30, 36, 43)
        for y in (-1.01, 1.01):
            line((0, y, 0), (0, y, -.48), (63, 72, 81), 2.)
        line((0, -b, -.02), (0, b, -.02), table, 2.)
        line((0, -b, 0), (0, b, 0), edge)
        line((0, 0, 0), (0, 0, NET_HEIGHT), net, 2.)
    return image


def _palette(rng, count):
    colors = []
    while len(colors) < count:
        rgb = [rng.randint(32, 205), rng.randint(32, 205), rng.randint(183, 251)]
        rng.shuffle(rgb)
        if all(sum((a - b)**2 for a, b in zip(rgb, prior)) >= 68**2 for prior in colors):
            colors.append(tuple(rgb))
    return colors


def _render(camera, pose, rival, rival_normal, ball, colors, decoy):
    image = _background(camera).copy()
    base, elbow, paddle, normal = pose
    base_px, elbow_px, paddle_px = (_project(point, camera) for point in (base, elbow, paddle))
    _line(image, base_px, elbow_px, (88, 96, 108), 2.)
    _line(image, elbow_px, paddle_px, (121, 129, 139), 2.)
    _ellipse(image, base_px, (3.5, 3.5), (88, 96, 108))
    _ellipse(image, elbow_px, (2., 2.), (121, 129, 139))
    rival_px = _project(rival, camera)
    _line(image, _project((0., 1.86, .11), camera), rival_px, (107, 115, 126), 2.)
    for center, axis, color in ((paddle_px, normal, colors[0]), (rival_px, rival_normal, colors[1])):
        radius = PADDLE_RADIUS * PIXELS_PER_METER
        lateral = axis[0] if camera == 0 else axis[2]
        _ellipse(image, center, (1.65 + radius * math.sqrt(max(0., 1. - axis[1]**2)),
                                 radius * math.sqrt(max(0., 1. - lateral**2))), color)
    if decoy is not None:
        _ellipse(image, _project(decoy, camera), (1.7, 1.7), colors[3])
    _ellipse(image, _project(ball, camera), (1.7, 1.7), colors[2])
    return image


def _strike(old_ball, ball, incoming, old_center, center, normal, dt):
    old_offset = tuple(a - b for a, b in zip(old_ball, old_center))
    offset = tuple(a - b for a, b in zip(ball, center))
    old_distance, distance = _dot(old_offset, normal), _dot(offset, normal)
    if not distance <= BALL_RADIUS < old_distance:
        return None
    fraction = (old_distance - BALL_RADIUS) / (old_distance - distance)
    contact_offset = tuple(a + fraction * (b - a) for a, b in zip(old_offset, offset))
    radial_squared = _dot(contact_offset, contact_offset) - BALL_RADIUS**2
    if radial_squared > (PADDLE_RADIUS + .35 * BALL_RADIUS)**2:
        return None
    body_velocity = tuple((b - a) / dt for a, b in zip(old_center, center))
    relative_normal_speed = _dot(tuple(v - b for v, b in zip(incoming, body_velocity)), normal)
    if relative_normal_speed >= 0.:
        return None
    impulse = -(1. + PADDLE_RESTITUTION) * relative_normal_speed
    outgoing = tuple(v + impulse * n for v, n in zip(incoming, normal))
    contact = tuple(a + fraction * (b - a) for a, b in zip(old_ball, ball))
    return contact, outgoing


class TT3Arena:
    """Two RGB cameras, joint-rate commands and signed public point reward.

    The overhead and side views share the horizontal court-depth axis. Width
    increases downward in the overhead view and height increases upward in the
    side view. Each case draws its own object colors independently of physics.

    World state and event details are available separately for evaluation. A policy
    receives the camera pixels, its outgoing command and the scalar reward.
    """

    def __init__(self, seed=0, *, view_seed=None, max_frames=300, variant='standard', distractor=False):
        if variant not in VARIANTS or type(max_frames) is not int or max_frames < 1:
            raise ValueError('expected a known variant and a positive frame budget')
        self.seed = seed
        self.view_seed = seed if view_seed is None else view_seed
        self.variant = variant
        self.distractor = distractor
        self.max_frames = max_frames
        self._point = -1
        self.reset(render=False)

    @property
    def paddle(self):
        return self._pose[2]

    @property
    def rival_normal(self):
        return self._rival_normal

    def reset(self, *, render=True):
        """Start the next seeded point and optionally return its camera images."""
        self._point += 1
        self._rng = random.Random(f'tt3/physics/v1/{self.seed}/{self._point}')
        view_rng = random.Random(f'tt3/appearance/v1/{self.view_seed}/{self._point}')
        self._colors = _palette(view_rng, 4 if self.distractor else 3)
        self._decoy_phases = tuple(view_rng.uniform(0., math.tau) for _ in range(3))
        self.ball, self.velocity = _opening(self._rng, self.variant)
        self.joints = INITIAL_JOINTS
        self._pose = arm_pose(self.joints)
        self.rival = (0., 1.27, .31)
        self._rival_normal = (0., -1., 0.)
        self._hitter = -1
        self._bounced = False
        self._rival_target_x = 0.
        self._rival_attempt = False
        self._rival_depth = 1.27
        self._rival_loft = .27
        self.frames = self.reward = self.contacts = self.legal_returns = self.rival_returns = 0
        self.events = []
        self.done = False
        return self.frame() if render else None

    def frame(self):
        decoy = None
        if self.distractor:
            t = self.frames / HZ
            px, py, pz = self._decoy_phases
            decoy = (.61 * math.sin(1.4 * t + px), 1.42 * math.sin(3.1 * t + py),
                     .45 + .31 * math.sin(4.7 * t + pz))
        return tuple(_render(camera, self._pose, self.rival, self._rival_normal, self.ball, self._colors, decoy)
                     for camera in range(2))

    def _event(self, kind, **details):
        self.events.append(dict(kind=kind, frame=self.frames + 1, ball=self.ball,
                                velocity=self.velocity, **details))

    def _point_to(self, winner, reason):
        self.reward = winner
        self.done = True
        self._event('point', winner=winner, reason=reason)

    def _opponent_try(self):
        self._rival_attempt = self._rng.random() < .88
        _, y, z = self.ball
        _, vy, vz = self.velocity
        to_bounce = (vz + math.sqrt(max(0., vz**2 + 2 * GRAVITY * (z - BALL_RADIUS)))) / GRAVITY
        self._rival_depth = _clamp(y + vy * to_bounce + self._rng.uniform(.24, .40), (.77, 1.43))
        self._rival_bias = self._rng.uniform(-.09, .09)
        self._rival_target_x = self._rng.choice((-1., 1.)) * self._rng.uniform(.17, .54)
        self._rival_loft = self._rng.uniform(-.038, .045)
        self._rival_target_depth = self._rng.uniform(.73, 1.01)

    def _move_rival(self, dt):
        x, y, z = self.ball
        vx, vy, vz = self.velocity
        if self._hitter == 1 and self._rival_attempt and vy > .35:
            swinging = self._bounced and y > self._rival_depth - .20
            horizon = max(0., (self.rival[1] - y) / (vy + 1.12) if swinging else (self._rival_depth - y) / vy)
            bounce_time = (vz + math.sqrt(max(0., vz**2 + 2 * GRAVITY * (z - BALL_RADIUS)))) / GRAVITY
            if not self._bounced and bounce_time < horizon:
                after = horizon - bounce_time
                target_vz = TABLE_RESTITUTION * (GRAVITY * bounce_time - vz) - GRAVITY * after
                target_z = BALL_RADIUS + target_vz * after + GRAVITY * after**2 / 2
            else:
                target_vz = vz - GRAVITY * horizon
                target_z = z + vz * horizon - GRAVITY * horizon**2 / 2
            strike_depth = self._rival_depth - .17 if swinging else self._rival_depth
            target = (_clamp(x + vx * horizon + self._rival_bias, (-.67, .67)),
                      strike_depth, _clamp(target_z, (.09, .64)))
            lateral = _clamp((self._rival_target_x - target[0]) * .19 - .14 * vx, (-.23, .23))
            # Estimate flight time using both inbound speed and the forward stroke.
            return_time = (self.rival[1] + self._rival_target_depth) / (.94 * vy + 2.13)
            return_vz = (BALL_RADIUS - target_z) / return_time + GRAVITY * return_time / 2
            loft = _clamp((return_vz - target_vz) / (2 * (vy + 1.12)) + self._rival_loft, (-.28, .52))
            self._rival_normal = _unit((lateral, -1., loft))
        else:
            target = (0., 1.27, .31)
        self.rival = tuple(a + _clamp(b - a, (-speed * dt, speed * dt)) for a, b, speed in
                           zip(self.rival, target, (1.75, 1.12, 1.2)))

    def _table_and_bounds(self, old_ball, dt):
        x, y, z = self.ball
        vx, vy, vz = self.velocity
        if (old_ball[1] < 0.) != (y < 0.):
            fraction = old_ball[1] / (old_ball[1] - y)
            net_x = old_ball[0] + fraction * (x - old_ball[0])
            net_z = old_ball[2] + fraction * (z - old_ball[2])
            if abs(net_x) <= TABLE_HALF_WIDTH + BALL_RADIUS and net_z <= NET_HEIGHT + BALL_RADIUS:
                self._point_to(-self._hitter, 'net')
                return
        if old_ball[2] >= BALL_RADIUS > z and vz < 0.:
            # Use the ballistic impact time so that launch speeds and bounce heights agree.
            prior_vz = vz + GRAVITY * dt
            time = (prior_vz + math.sqrt(prior_vz**2 + 2 * GRAVITY * (old_ball[2] - BALL_RADIUS))) / GRAVITY
            bx, by = old_ball[0] + vx * time, old_ball[1] + vy * time
            if abs(bx) <= TABLE_HALF_WIDTH and abs(by) <= TABLE_HALF_LENGTH:
                side = 1 if by < 0. else -1
                self._event('bounce', side=side, location=(bx, by, BALL_RADIUS))
                if side == self._hitter:
                    self._point_to(-self._hitter, 'own_court')
                    return
                if self._bounced:
                    self._point_to(self._hitter, 'second_bounce')
                    return
                self._bounced = True
                if self._hitter == 1:
                    self.legal_returns += 1
                    self._event('legal_return', side=1, location=(bx, by, BALL_RADIUS))
                elif self.contacts:
                    self.rival_returns += 1
                    self._event('legal_return', side=-1, location=(bx, by, BALL_RADIUS))
                upward = TABLE_RESTITUTION * (GRAVITY * time - prior_vz)
                rest = dt - time
                self.ball = (x, y, BALL_RADIUS + upward * rest - GRAVITY * rest**2 / 2)
                self.velocity = (vx, vy, upward - GRAVITY * rest)
                return
        if abs(x) > 1.16 or abs(y) > 2.10 or z < -.28 or z > 1.85:
            self._point_to(self._hitter if self._bounced else -self._hitter, 'out')

    def _advance_arm(self, delta, dt):
        self.joints = tuple(_clamp(q + dq, limits) for q, dq, limits in
                            zip(self.joints, delta, JOINT_LIMITS))
        self._pose = arm_pose(self.joints)

    def step(self, action, *, render=True):
        """Advance one frame using normalized yaw, shoulder and elbow rates."""
        if self.done:
            raise RuntimeError('reset after a point ends')
        command = np.asarray(action, dtype=float)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError('expected three finite joint commands')
        if (np.abs(command) > 1.).any():
            raise ValueError('joint commands must lie in [-1, 1]')
        dt = 1. / (HZ * SUBSTEPS)
        delta = tuple(value * speed * dt for value, speed in zip(command.tolist(), JOINT_RATES))
        prior_events = len(self.events)
        for _ in range(SUBSTEPS):
            old_ball, old_center, old_rival = self.ball, self.paddle, self.rival
            self._advance_arm(delta, dt)
            self._move_rival(dt)
            x, y, z = old_ball
            vx, vy, vz = self.velocity
            self.ball = (x + vx * dt, y + vy * dt, z + vz * dt - GRAVITY * dt**2 / 2)
            self.velocity = (vx, vy, vz - GRAVITY * dt)
            self._table_and_bounds(old_ball, dt)
            if self.done:
                break
            side = -self._hitter
            hit = None
            if side == 1 and self.ball[1] < -.55 and self.velocity[1] < 0.:
                hit = _strike(old_ball, self.ball, self.velocity, old_center, self.paddle, self._pose[3], dt)
            if side == -1 and self.ball[1] > .58 and self.velocity[1] > 0. and self._rival_attempt and self._bounced:
                hit = _strike(old_ball, self.ball, self.velocity, old_rival, self.rival, self._rival_normal, dt)
            if hit is not None:
                self.ball, outgoing = hit
                if side == 1:
                    self.contacts += 1
                self._event('contact', side=side, outgoing=outgoing)
                if not self._bounced:
                    self._point_to(-side, 'volley')
                    break
                self.velocity = outgoing
                self._hitter, self._bounced = side, False
                if side == 1:
                    self._opponent_try()
        if self.frames + 1 >= self.max_frames and not self.done:
            self._point_to(0, 'timeout')
        self.frames += 1
        info = dict(contacts=self.contacts, legal_returns=self.legal_returns, rival_returns=self.rival_returns,
                    points_for=int(self.reward > 0), points_against=int(self.reward < 0),
                    events=tuple(self.events[prior_events:]))
        return self.frame() if render else None, self.reward, self.done, info

    def point_result(self):
        """Return a JSON-ready score and physical-return counts for a finished point."""
        if not self.done:
            raise RuntimeError('finish the point before requesting its result')
        return dict(contacts=self.contacts, legal_returns=self.legal_returns, rival_returns=self.rival_returns,
                    points_for=int(self.reward > 0), points_against=int(self.reward < 0),
                    frames=self.frames, reason=self.events[-1]['reason'])
