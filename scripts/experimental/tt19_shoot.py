"""Search learned optical paddle trajectories for complete table strokes."""
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .tt19_flight import optical_start


WIDTH = 5
RADII = np.triu_indices(WIDTH)
BOUND = 3.80
HORIZON = 20
BALL_RADIUS = .020
DISK_RADIUS = .076
HALF_THICKNESS = .006
COURT = (.424, .80)
CLEARANCE = .180


def filtered(commands, kernel):
    return np.cumsum(commands, axis=-2) + lfilter(kernel - 1., [1.], commands, axis=-2)


def quadratic(q):
    first, second = RADII
    return np.concatenate((q, q[..., first] * q[..., second]), axis=-1)


def observed_pose(tracker):
    if tracker.actor is None or not tracker.actor.observed:
        return None
    normal = tracker.actor_normal
    return np.r_[tracker.actor.position, [np.nan, np.nan] if normal is None else normal[[0, 2]]]


@dataclass(frozen=True)
class OpticalBody:
    kernel: np.ndarray
    lag: int
    home: np.ndarray
    weights: np.ndarray

    def coordinates(self, commands):
        commands = np.asarray(commands)[..., :WIDTH]
        zero = np.zeros(commands.shape[:-2] + (self.lag + 1, WIDTH))
        if not commands.shape[-2]:
            return zero[..., :1, :]
        trajectory = filtered(commands, self.kernel)
        return np.concatenate((zero, trajectory), axis=-2)[..., :commands.shape[-2] + 1, :]

    def pose(self, q):
        return self.home + quadratic(q) @ self.weights

    def jacobian(self, q):
        matrix = self.weights[:WIDTH].copy()
        first, second = RADII
        np.add.at(matrix, first, self.weights[WIDTH:] * q[second, None])
        np.add.at(matrix, second, self.weights[WIDTH:] * q[first, None])
        return matrix.T

    def state(self):
        return dict(kernel=self.kernel.tolist(), lag=self.lag, home=self.home.tolist(), weights=self.weights.tolist())

    @classmethod
    def restore(cls, data):
        return cls(np.asarray(data['kernel']), data['lag'], np.asarray(data['home']), np.asarray(data['weights']))

    @classmethod
    def fit(cls, traces, kernel, lag):
        shell = cls(np.asarray(kernel), lag, np.zeros(5), np.zeros((20, 5)))
        poses = np.concatenate([np.asarray(row['poses'], dtype=float) for row in traces])
        samples = np.concatenate([quadratic(shell.coordinates(row['actions'])) for row in traces])
        design = np.column_stack((np.ones(len(samples)), samples))
        coefficients = np.zeros((design.shape[1], 5))
        penalty = np.diag(np.r_[.003, np.full(WIDTH, .28), np.full(len(RADII[0]), 3.1)])
        for axis in range(5):
            valid = np.isfinite(poses[:, axis])
            if axis < 3 and valid.sum() < 24:
                raise ValueError('not enough RGB motion samples to fit the tool position')
            if not np.any(valid):
                continue
            features, target = design[valid], poses[valid, axis]
            precision = np.ones(len(features))
            for _ in range(3):
                weighted = precision[:, None] * features
                fit = np.linalg.solve(features.T @ weighted + penalty, weighted.T @ target)
                precision = (1 + ((features @ fit - target) / (.031 if axis < 3 else .15))**2)**-.5
            coefficients[:, axis] = fit
        return cls(shell.kernel, lag, coefficients[0], coefficients[1:])


def qualification(body, rows):
    errors, angles = [], []
    for row in rows:
        actual = np.asarray(row['poses'], dtype=float)
        predicted = body.pose(body.coordinates(row['actions']))
        errors.extend(np.linalg.norm(predicted[:, :3] - actual[:, :3], axis=1))
        good = np.isfinite(actual[:, 3:]).all(axis=1)
        angles.extend(np.rad2deg(np.arccos(np.clip(np.sum(_normals(predicted[good]) * _normals(actual[good]), axis=1), -1., 1.))))
    errors = np.asarray(errors)
    errors = errors[np.isfinite(errors)]
    return dict(position_samples=len(errors), median_m=float(np.median(errors)), p90_m=float(np.quantile(errors, .9)),
                orientation_samples=len(angles), median_orientation_deg=None if not angles else float(np.median(angles)))


def _normals(poses):
    x, z = np.moveaxis(poses[..., 3:], -1, 0)
    size = np.maximum(1., np.sqrt(x*x + z*z) / .985)
    x, z = x / size, z / size
    return np.stack((x, np.sqrt(np.maximum(0., 1. - x*x - z*z)), z), axis=-1)


def _unit(vectors):
    return vectors / np.maximum(1e-8, np.linalg.norm(vectors, axis=-1, keepdims=True))


def _surface(poses, points, speeds):
    """Sample moving sphere-to-disk distances and preserve the first contact per tape."""
    interval = np.repeat(np.arange(HORIZON), 4)
    f = (np.tile(np.arange(1, 5), HORIZON) * .25)[:, None]
    start = poses[:, interval, :3]
    motion = poses[:, interval + 1, :3] - start
    centers = start + f * motion
    blade0, blade1 = _normals(poses[:, interval]), _normals(poses[:, interval + 1])
    normals = _unit(blade0 + f * (blade1 - blade0))
    angular = np.cross(blade0, blade1)
    ball = points[interval] + f * (points[interval + 1] - points[interval])
    incoming = speeds[interval] + f * (speeds[interval + 1] - speeds[interval])
    delta = ball - centers
    signed = np.sum(delta * normals, axis=-1, keepdims=True)
    radial = delta - signed * normals
    radial_size = np.linalg.norm(radial, axis=-1, keepdims=True)
    edge = np.maximum(0., radial_size - DISK_RADIUS)
    surface = np.sign(signed) * normals * np.maximum(0., abs(signed) - HALF_THICKNESS) + _unit(radial) * edge
    separation = np.linalg.norm(surface, axis=-1) - BALL_RADIUS
    contact_axis = _unit(surface)
    tiny = np.linalg.norm(surface, axis=-1) < 1e-7
    contact_axis[tiny] = normals[tiny]
    velocity = motion + np.cross(angular, radial)
    # The sweep must approach the surface. Grazing or separating contacts
    # remain in the geometric distance audit, but cannot supply an impulse.
    approach = np.sum((incoming - velocity) * contact_axis, axis=-1) < -.004
    actual = (separation <= .001) & approach
    hit = np.any(actual, axis=1)
    first = np.argmax(actual, axis=1)
    alternative = np.argmin(separation + .007 * (~approach), axis=1)
    chosen = np.where(hit, first, alternative)
    rows = np.arange(len(poses))
    return (hit, interval[chosen] + f[chosen, 0], ball[chosen], incoming[chosen], contact_axis[rows, chosen],
            velocity[rows, chosen], separation[rows, chosen], radial_size[rows, chosen, 0], approach[rows, chosen])


def _landings(collision, incoming, surface, blade_velocity, law, response, serving):
    tangent = _unit(np.array((0., 0., 1.)) - surface[..., 2:] * surface)
    sideways = np.cross(surface, tangent)
    relative = incoming - blade_velocity
    local = np.stack((np.sum(relative * surface, axis=-1), np.sum(relative * tangent, axis=-1)), axis=-1)
    impulse = local @ response
    out = incoming + impulse[:, :1] * surface + impulse[:, 1:] * tangent
    out -= .19 * np.sum(relative * sideways, axis=-1, keepdims=True) * sideways
    global_y = np.maximum(.003, out[:, 1])
    gx, gy, gz = collision.T

    def arc(height, vertical, y, vy):
        speed = np.maximum(.003, vy)
        t = (vertical + np.sqrt(np.maximum(0., vertical**2 + 2. * law.gravity * (height - BALL_RADIUS)))) / law.gravity
        distance = np.log1p(t * law.drag * speed) / law.drag
        at_net = np.expm1(np.clip(-y * law.drag, -1., 1.)) / (law.drag * speed)
        net = height + at_net * (vertical - .5 * law.gravity * at_net)
        return y + distance, net, t

    first, height, elapsed = arc(gz, out[:, 2], gy, global_y)
    distance = first
    x = gx + elapsed * out[:, 0]
    first_cost = np.zeros(len(out))
    if serving:
        first_cost = 5.0 * (np.maximum(0., first + .08) + np.maximum(0., -.76 - first))
        second_vy = global_y * law.tangent / (1. + elapsed * law.drag * global_y)
        second_vz = (law.gravity * elapsed - out[:, 2]) * law.restitution
        distance, height, second_elapsed = arc(BALL_RADIUS, second_vz, first, second_vy)
        x += second_elapsed * out[:, 0] * law.tangent
    cost = (.45 * abs(distance - .40) + .48 * abs(x)
            + 5.8 * (np.maximum(0., distance - .76) + np.maximum(0., .045 - distance))
            + 6.5 * np.maximum(0., CLEARANCE - height) + 5.4 * np.maximum(0., abs(x) - .388)
            + 3.5 * np.maximum(0., .025 - out[:, 1]) + first_cost)
    return cost, np.column_stack((x, distance)), height, out, first


def _candidates(rng, old, ball, speed, model, shift, serving, count):
    q = model.coordinates(old)[-1]
    pose = model.pose(q)
    position, normal = pose[:3], _normals(pose)
    jac = model.jacobian(q)
    weight = np.array([1., 1., 1., .10, .11])
    design = weight[:, None] * jac
    inverse = np.linalg.solve(design.T @ design + .00045 * np.eye(WIDTH), design.T)
    eligible = ((ball[:, 1] < -.79) & (ball[:, 1] >= -1.14) & (ball[:, 2] >= .045) & (ball[:, 2] < .43))
    eligible &= (speed[:, 2] < -.012) if serving else (speed[:, 1] < -.013)
    eligible[0:2] = False
    if not eligible.any():
        return None
    times = rng.choice(np.flatnonzero(eligible), count)
    pitch = rng.uniform(.08, .78 if serving else .65, count)
    yaw = np.clip(-3.0 * speed[times, 0] - .28 * ball[times, 0] + rng.normal(0., .11, count), -.5, .5)
    target_center = ball[times] - (BALL_RADIUS + HALF_THICKNESS) * np.column_stack((yaw, np.cos(pitch), np.sin(pitch)))
    target = np.column_stack((target_center, yaw, np.sin(pitch)))
    present = np.r_[position + shift[:3], (normal + np.array((shift[3], 0., shift[4])))[[0, 2]]]
    delta = ((target - present) * weight) @ inverse.T
    move = np.zeros((count, WIDTH))
    move[:] = np.array((0., 1., rng.uniform(-.30, .45), 0., 0.)) @ inverse.T
    move /= np.maximum(.015, np.linalg.norm(move, axis=1, keepdims=True))
    second = move * rng.uniform(.22, .86, (count, 1)) + rng.normal(0., .32, (count, WIDTH))
    burst = rng.integers(2, 7, size=count)
    onset = np.maximum(0, times - burst)
    early = (delta - second * np.minimum(times, burst)[:, None]) / np.maximum(2, onset)[:, None]
    early += rng.normal(0., .19, (count, WIDTH))
    # Contiguous two-stage sweeps make speed and approach direction explicit.
    ticks = np.arange(HORIZON)[None, :]
    tapes = np.where((ticks < onset[:, None])[..., None], early[:, None, :], second[:, None, :])
    tapes = np.where((ticks < (times + 2)[:, None])[..., None], tapes, 0.)
    jittered = count // 5
    if jittered:
        tapes[-jittered:] += rng.normal(0., .34, (jittered, 1, WIDTH))
    return np.clip(tapes, -.98, .98)


def plan_shot(model, history, shift, estimate, law, response, serving, rng, count=540):
    points, speeds = law.trajectory(*estimate, HORIZON, bounds=COURT)
    tapes = _candidates(rng, history, points, speeds, model, shift, serving, count)
    if tapes is None:
        return None
    cumulative = np.cumsum(tapes, axis=1) + np.sum(history, axis=0)
    bounded = np.clip(cumulative, -BOUND, BOUND)
    tapes[:, 0] = bounded[:, 0] - np.sum(history, axis=0)
    tapes[:, 1:] = np.diff(bounded, axis=1)
    prior = np.broadcast_to(history, (count, *history.shape))
    paths = model.pose(model.coordinates(np.concatenate([prior, tapes], axis=1))[:, -HORIZON-1:]) + shift
    hit, eta, ball, incoming, face, rates, separation, radial, approach = _surface(paths, points, speeds)
    loss, land, net, outgoing, first = _landings(ball, incoming, face, rates, law, response, serving)
    legal_depth = ((ball[:, 1] < -COURT[1]) & (incoming[:, 2] < -.008)) if serving else (ball[:, 1] < -.74)
    # A missed proposal still supplies a useful approach gradient. Reserve a
    # larger penalty for makeable contacts with a bad face or landing arc.
    loss += 9. * np.maximum(0., separation) + .30 * (~hit) + .24 * (~approach) + 1.0 * (~legal_depth)
    loss += 2.9 * np.maximum(0., radial - .050) + .0007 * np.sum(tapes**2, axis=(1, 2))
    loss += 6. * np.maximum(0., .045 - ball[:, 2])
    candidate = int(np.argmin(loss))
    return tapes[candidate], dict(eta=float(eta[candidate]), predicted_contact=bool(hit[candidate]),
                                  legal_depth=bool(legal_depth[candidate]), separation=float(separation[candidate]),
                                  objective=float(loss[candidate]), ball=ball[candidate].tolist(),
                                  blade_speed=rates[candidate].tolist(), face=face[candidate].tolist(),
                                  landing=land[candidate].tolist(), net=float(net[candidate]),
                                  first_bounce=float(first[candidate]), candidates=count)


class ShootingPlayer:
    def __init__(self, tracker, model, law, response, server, seed, *, count=540):
        self.tracker, self.model, self.law, self.response = tracker, model, law, response
        self.server = server
        self.rng = np.random.default_rng(seed)
        self.count = count
        self.history, self.shots, self.optical = [], [], []
        self.commands = np.empty((0, WIDTH))
        self.plan = None
        self.shift = np.zeros(5)
        self.served = False
        self.previous = None
        self.cooldown = 0
        self.tick = -1
        self.motion = np.zeros(WIDTH)

    def observe(self, images):
        self.tick += 1
        if images is None:
            return
        self.tracker.update(images)
        ball = self.tracker.ball
        sight = None if ball is None or not ball.observed else tuple(map(float, ball.position))
        self.observe_measurements(sight, observed_pose(self.tracker))

    def observe_measurements(self, sight, pose):
        """Update from the observations retained in the point receipt."""
        if sight is not None and abs(sight[0]) > .75:
            sight = None
        self.history.append((self.tick, sight))
        if pose is not None:
            expected = self.model.pose(self.model.coordinates(self.commands)[-1])
            finite = np.isfinite(pose)
            weight = .63 if self.tick == 0 else .45
            self.shift[finite] = (1. - weight) * self.shift[finite] + weight * np.clip(pose[finite] - expected[finite], -.12, .12)
        self.optical.append(dict(tick=self.tick, ball=sight, blade=None if pose is None else [
            float(value) if np.isfinite(value) else None for value in pose]))

    def choose(self):
        estimate = optical_start(self.history, self.law.gravity)
        command = np.zeros(WIDTH)
        if estimate is not None:
            point, speed = estimate
            if (self.previous is not None and point[1] < -.55 and speed[1] > .028
                    and (self.previous[1] < -.016 or (self.server == 1 and not self.served))):
                self.plan = None
                self.served = True
                self.cooldown = self.tick + 8
            self.previous = speed
            serving = self.server == 1 and not self.served
            attacking = abs(speed[1]) < .025 and self.tick < 27 if serving else speed[1] < -.016
            if attacking and self.tick >= self.cooldown and (self.plan is None or self.tick % 3 == 0 and self.plan[0] - self.tick > 3):
                forecast = plan_shot(self.model, self.commands, self.shift, estimate, self.law, self.response,
                                     serving, self.rng, self.count)
                if forecast is not None:
                    tape, detail = forecast
                    if detail['separation'] < .29:
                        self.plan = (self.tick + detail['eta'], self.tick, tape)
                        self.shots.append(dict(tick=self.tick, serving=serving, **detail))
        if self.plan is not None:
            contact, started, tape = self.plan
            index = self.tick - started
            if index < len(tape) and self.tick <= contact + 3:
                command = tape[index]
            else:
                self.plan = None
        elif self.tick >= self.cooldown and self.tick > 5:
            command = np.clip(-self.motion * .26, -.30, .30)
        self.commands = np.concatenate((self.commands, command[None, :]), axis=0)
        self.motion += command
        return np.r_[command, 0.]
