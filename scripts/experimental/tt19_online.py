"""Play optical table tennis with a learned inverse motor response."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .tt7_flight import FlightMemory, nominal_exit
from .tt19_flight import FlightLaw, optical_start
from .tt19_online_body import ORIENTATION, blade_pose


BALL = .020


@dataclass(frozen=True)
class Court:
    half_length: float
    half_width: float
    net_clearance: float


def launch_options(position, incoming, pace, court, law, response, serving, minimum_rate=.015):
    """Choose a feasible blade velocity and direction by simulating bounce order."""
    angles, rates = np.meshgrid(np.linspace(.10, .74, 25), np.linspace(minimum_rate, max(minimum_rate, pace), 7))
    angles, rates = angles.ravel(), rates.ravel()
    yaw = np.clip(-2.3 * incoming[0] - .26 * position[0], -.35, .35)
    sine, cosine = np.sin(angles), np.cos(angles)
    normal = np.stack([np.sin(yaw) * cosine, np.cos(yaw) * cosine, sine], axis=1)
    tangent = np.stack([-np.sin(yaw) * sine, -np.cos(yaw) * sine, cosine], axis=1)
    relative = incoming - np.column_stack([np.zeros_like(rates), rates, np.zeros_like(rates)])
    local = np.column_stack([np.sum(normal * relative, axis=1), np.sum(tangent * relative, axis=1)])
    impulse = local @ response
    outbound = incoming + normal * impulse[:, :1] + tangent * impulse[:, 1:]
    fwd = np.maximum(.005, outbound[:, 1])
    rise = outbound[:, 2]

    def ground_time(height, vertical):
        return (vertical + np.sqrt(np.maximum(0., vertical**2 + 2 * law.gravity * height))) / law.gravity

    def depth(time, speed):
        return np.log1p(np.maximum(0., time) * law.drag * speed) / law.drag

    before = ground_time(position[2] - BALL, rise)
    first = position[1] + depth(before, fwd)
    base_y, base_z = position[1], position[2]
    if serving:
        base_y, base_z = first, BALL
        rise = -(rise - law.gravity * before) * law.restitution
        fwd = fwd * law.tangent / (1. + before * law.drag * fwd)
    to_net = np.expm1(law.drag * -base_y) / (law.drag * fwd)
    net_height = base_z + rise * to_net - .5 * law.gravity * to_net**2
    to_far = ground_time(base_z - BALL, rise)
    final = base_y + depth(to_far, fwd)
    final_x = position[0] + (before + (to_far if serving else 0.)) * outbound[:, 0]
    score = np.abs(final - .56 * court.half_length)
    score += 7. * np.maximum(0., court.net_clearance + .033 - net_height)
    score += 4. * np.maximum(0., -final + .06)
    score += 4. * np.maximum(0., final - court.half_length + .045)
    score += 3. * np.maximum(0., abs(final_x) - court.half_width + .03)
    score += 5. * np.maximum(0., .012 - outbound[:, 1])
    score += .24 * abs(angles - .29) + .65 * rates
    if serving:
        score += 5. * np.maximum(0., first + .07)
        score += 5. * np.maximum(0., -court.half_length + .025 - first)
    picked = np.argmin(score)
    return float(angles[picked]), float(yaw), float(rates[picked]), float(score[picked])


@dataclass
class Stroke:
    tick: float
    center: np.ndarray
    incoming: np.ndarray
    pitch: float
    yaw: float
    rate: float
    role: str

    @property
    def normal(self):
        return np.array([np.sin(self.yaw) * np.cos(self.pitch), np.cos(self.yaw) * np.cos(self.pitch),
                         np.sin(self.pitch)])

    def pose(self, tick, pitch_lead=0.):
        normal = self.normal
        center = self.center - (BALL + .005) * normal
        center[1] += np.clip((tick - self.tick) * self.rate, -.039, .079)
        if pitch_lead:
            pitch = np.clip(self.pitch + pitch_lead, -.18, .99)
            orientation = [np.sin(pitch), np.sin(self.yaw) * np.cos(pitch)]
        else:
            orientation = normal[[2, 0]]
        return np.r_[center, ORIENTATION * np.asarray(orientation)]


class OpticalPlayer:
    def __init__(self, body, tracker, court, law, response, *, explore=True):
        self.body = body
        self.tracker_factory = tracker
        self.court = court
        self.law = law
        self.response = response
        self.explore = explore
        self.bounce_samples = []
        self.shot_samples = 0
        self.registration = dict(serve=0., receive=0.)
        self.registration_count = dict(serve=0, receive=0)

    def reset(self):
        self.body.reset()
        self.tracker = self.tracker_factory()
        self.trace = []
        self.tick = -1
        self.history = []
        self.serving = False
        self.served = False
        self.plan = self.pending = None
        self.ready = None
        self.last_launch = -20
        self.known_returns = 0
        self.attempt = {}

    def _flight(self):
        fit = optical_start(self.history, self.law.gravity)
        if fit is not None:
            return fit
        good = [(tick, pose) for tick, pose in self.history[-5:] if pose is not None]
        if len(good) < 3 or self.tick - good[-1][0] > 3:
            return None
        previous = optical_start(good, self.law.gravity)
        if previous is None:
            return None
        points, speeds = self._project(*previous, self.tick - good[-1][0])
        return points[-1], speeds[-1]

    def _project(self, point, velocity, horizon):
        return self.law.trajectory(point, velocity, horizon, bounds=(self.court.half_width, self.court.half_length))

    def _bounce_sample(self):
        if len(self.history) < 4:
            return
        times, seen = zip(*self.history[-4:])
        if any(pose is None for pose in seen) or times[-1] - times[0] != 3:
            return
        z = np.asarray(seen)[:, 2]
        a, b, c = np.diff(z)
        if a < -.035 and c > .022 and min(z) < .074 and a < b < c:
            self.bounce_samples.append(float(np.clip((c + .5 * self.law.gravity) /
                                                      (-a + .5 * self.law.gravity), .44, 1.10)))

    def _prepare(self, state):
        if state is None or self.ready is None:
            return None
        point, velocity = state
        if (self.tick <= 12 and velocity[2] > .035 and -.20 < point[1] - self.ready[1] < .24
                and abs(velocity[1]) < .024):
            self.serving = True
        if self.serving and not self.served:
            if self.tick > 30 or velocity[1] > .027:
                self.served = True
                return None
            z = float(np.clip(self.ready[2] + .054, .167, .36))
            discriminant = velocity[2]**2 + 2. * self.law.gravity * (point[2] - z)
            if discriminant <= 0.:
                return None
            time = (velocity[2] + np.sqrt(discriminant)) / self.law.gravity
            if time < -.6 or time > 25.:
                return None
            positions = (point[None] + time * velocity[None]).copy()
            velocities = velocity[None].copy()
            positions[0, 2] = z
            velocities[0, 2] = -np.sqrt(discriminant)
            times = np.array([time])
            role = 'serve'
        elif velocity[1] < -.018:
            positions, velocities = self._project(point, velocity, 26)
            delta = positions[:, 1] - self.ready[1]
            choices = np.flatnonzero((delta > -.065) & (delta < .082) & (positions[:, 2] > .063)
                                     & (positions[:, 2] < self.ready[2] + .16) & (velocities[:, 1] < -.012))
            if not len(choices):
                return None
            times = choices
            positions, velocities = positions[choices], velocities[choices]
            role = 'receive'
        else:
            return None
        q = self.body.q[self.body.lag]
        jac = self.body.jacobian(q)
        live = self.body.live_pose()
        serve_cap = .068 if self.body.solver in ('clipped', 'register') else .030
        pace = np.clip(.55 * np.sum(abs(jac[1])), .023, serve_cap if role == 'serve' else .068)
        reach = np.linalg.solve(jac[:3] @ jac[:3].T + .00040 * np.eye(3), jac[:3])
        targets = positions - np.array([0., .041, 0.])
        delta = targets - live[:3]
        motion = np.max(abs(delta @ reach), axis=1)
        duration = times + 1.4
        ahead = -.018 if self.body.solver == 'drive' else .026
        prior = .52 * abs(positions[:, 2] - (self.ready[2] + .026)) + .82 * abs(
            positions[:, 1] - self.ready[1] - ahead)
        score = prior + .026 * np.maximum(0., motion + self.body.lag + 2.3 - duration)
        order = np.argsort(score)[:5]
        candidates = []
        for index in order:
            horizon = times[index]
            center, incoming = positions[index], velocities[index]
            threshold = .039 if self.body.solver == 'drive' and role == 'receive' else .015
            angle, yaw, rate, landing = launch_options(center, incoming, pace, self.court, self.law,
                                                       self.response.response_matrix, role == 'serve', threshold)
            rotation = np.array([np.sin(angle), np.sin(yaw) * np.cos(angle)]) * ORIENTATION
            loss = score[index] + .035 * landing + .5 * np.linalg.norm(rotation - self.ready[3:])
            candidates.append((loss, Stroke(float(self.tick + horizon), center, incoming, angle, yaw, rate, role)))
        return min(candidates, key=lambda item: item[0])[1]

    def _carry_plan(self, proposed):
        if proposed is None:
            if self.plan is not None and -1.5 <= self.tick - self.plan.tick <= 4.:
                return
            self.plan = None
            return
        current = self.plan
        if current is None or current.role != proposed.role or self.tick > current.tick + 3.:
            self.plan = proposed
            return
        amount = .34 if abs(proposed.tick - current.tick) <= 3. else .72
        if current.tick - self.tick < 2. and proposed.tick - self.tick < 4.:
            amount = min(.24, amount)
        self.plan = Stroke((1 - amount) * current.tick + amount * proposed.tick,
                           (1 - amount) * current.center + amount * proposed.center,
                           proposed.incoming, (1 - amount) * current.pitch + amount * proposed.pitch,
                           (1 - amount) * current.yaw + amount * proposed.yaw,
                           (1 - amount) * current.rate + amount * proposed.rate, proposed.role)

    def _measure_departure(self, state, pose):
        if self.pending is None or state is None or self.tick - self.pending.tick < 3:
            return
        if self.tick - self.pending.tick > 12:
            self.pending = None
            return
        point, velocity = state
        if velocity[1] < .036 or not 0.03 < point[1] - self.pending.center[1] < .90:
            return
        strike = self.pending
        seen = [row for row in self.trace if row['blade'] is not None and abs(row['tick'] - strike.tick) <= 2.5]
        if len(seen) >= 2:
            start, end = seen[0], seen[-1]
            forward = (end['blade'][1] - start['blade'][1]) / (end['tick'] - start['tick'])
            measured = min(seen, key=lambda row: abs(row['tick'] - strike.tick))
            angle = strike.pitch if measured['blade_pitch'] is None else measured['blade_pitch']
            if np.linalg.norm(np.asarray(measured['blade']) - strike.center) < .15:
                incoming = strike.incoming[1:]
                self.response.sample((incoming, angle, forward), velocity[1:],
                                     nominal_exit(incoming, angle, forward, .80))
                self.shot_samples += 1
        self.served = self.served or strike.role == 'serve'
        self.known_returns += strike.role == 'receive'
        self.last_launch = self.tick
        self.plan = self.pending = None

    def update(self, frames, preceding=None):
        self.tick += 1
        self.body.advance(preceding)
        self.tracker.update(frames)
        pose = blade_pose(self.tracker)
        self.body.measure(pose)
        if self.ready is None and self.body.home is not None:
            self.ready = np.nan_to_num(self.body.home.copy())
            self.ready[3:] *= .55
        ball = self.tracker.ball
        sight = None if ball is None or not ball.observed else np.asarray(ball.position)
        self.history.append((self.tick, sight))
        self._bounce_sample()
        state = self._flight()
        self._measure_departure(state, pose)
        self._carry_plan(self._prepare(state))
        plan = self.plan
        if plan is not None and -1.7 < plan.tick - self.tick < 1.7 and self.tick - self.last_launch > 6:
            self.pending = plan
        if plan is None:
            def target(_):
                return self.ready
            deadline = None
        else:
            bias = self.registration[plan.role]
            def target(horizon):
                return plan.pose(self.tick + horizon, bias)
            deadline = plan.tick - self.tick
            pitch = self.tracker.actor_pitch
            if self.body.solver == 'register' and abs(deadline) <= 2.0 and pose is not None and pitch is not None:
                distance = float(np.linalg.norm(pose[:3] - plan.center))
                if distance < .17:
                    quality = float(abs(deadline) + 9. * distance)
                    old = self.attempt.get(plan.role)
                    if old is None or quality < old['quality']:
                        self.attempt[plan.role] = dict(quality=quality, intended=plan.pitch, measured=pitch,
                                                       tick=self.tick, deadline=float(deadline))
        action = self.body.control(target, deadline, explore=self.explore, fixed=(-1,),
                                   strike_rate=0. if plan is None else plan.rate)
        self.trace.append(dict(tick=self.tick, ball=None if sight is None else sight.tolist(),
                               blade=None if pose is None else pose[:3].tolist(),
                               blade_pitch=self.tracker.actor_pitch,
                               blade_yaw=getattr(self.tracker, 'actor_yaw', None),
                               horizon=deadline, estimate=None if state is None else [row.tolist() for row in state],
                               goal=None if self.ready is None else target(0).tolist(),
                               role=None if plan is None else plan.role, lag=self.body.lag,
                               model_error=self.body.error, samples=self.body.samples))
        return action

    def finish(self, reward):
        changes = {}
        for role, sample in self.attempt.items():
            prior = self.registration[role]
            residual = np.clip(sample['intended'] - sample['measured'], -.37, .37)
            gain = .15 if reward > 0 else .55
            self.registration[role] = float(np.clip(.92 * prior + gain * residual, -.31, .47))
            self.registration_count[role] += 1
            changes[role] = dict(sample, prior=prior, next=self.registration[role])
        result = dict(exploration_rate_frames=self.body.exploration, visually_returned=self.known_returns,
                      shots_observed=self.shot_samples)
        if self.body.solver == 'register':
            result['pitch_registration'] = changes
        return result
