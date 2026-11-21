"""Choose a reply strike depth using an RGB-calibrated forecast of blade motion."""
from dataclasses import asdict, dataclass
from math import cos, sin

import numpy as np

from .tt5_player import (
    BALL_RADIUS, DRAG, GRAVITY, NET_CLEARANCE, TABLE_END, TABLE_HALF_WIDTH,
    _prediction,
)
from .tt6_bounded import _clipped_plan
from .tt9_precontact import FACE_DISTANCE
from .tt9_precontact_registration import RegistrationPlayer


DEPTH_OFFSETS = (0., -.15, -.075, .075, .15, .225)
PITCH_OFFSETS = (0., -.09, .09)
MIN_IMPROVEMENT = .035


@dataclass
class Strike:
    time: float
    point: np.ndarray
    incoming: np.ndarray
    pitch: float

    def target(self, remaining, rate):
        center = self.point - FACE_DISTANCE * np.array([0., cos(self.pitch), sin(self.pitch)])
        center[1] -= np.clip(rate * remaining, -.107, .145)
        center[2] = max(.054, center[2])
        return center


@dataclass
class Forecast:
    cost: float
    time: float
    ball: tuple
    blade_velocity: tuple
    pitch: float
    miss: float
    clearance: float
    bounce_depth: float | None


def _flight_cost(position, incoming, blade_speed, pitch, restitution, correction):
    """Evaluate net clearance and first bounce using the usual depth-drag prior."""
    normal = np.array([0., cos(pitch), sin(pitch)])
    tangent = np.array([0., -normal[2], normal[1]])
    relative = incoming - blade_speed
    closing = -relative @ normal
    outgoing = incoming + (1. + restitution) * closing * normal
    outgoing -= .23 * ((relative @ tangent) * tangent + [relative[0], 0., 0.])
    outgoing[1:] += correction
    if outgoing[1] <= .02:
        return 1. + max(0., .02 - outgoing[1]), -1., None

    speed = outgoing[1]
    depths = np.array([0., .24, .75, 1.22, TABLE_END])
    travel = np.expm1(DRAG * (depths - position[1])) / (DRAG * speed)
    decay = -np.expm1(-DRAG * speed * travel) / (DRAG * speed)
    height = position[2] + (outgoing[2] + GRAVITY / (DRAG * speed)) * decay
    height -= GRAVITY / (DRAG * speed) * travel
    lateral = position[0] + outgoing[0] * travel
    clearance = height[0] - NET_CLEARANCE
    crossed = np.flatnonzero(height[1:] < BALL_RADIUS)
    bounce = None
    if len(crossed) and height[0] > BALL_RADIUS:
        i = crossed[0]
        bounce = float(np.interp(BALL_RADIUS, height[i:i + 2][::-1], depths[i:i + 2][::-1]))
    cost = (2.4 * max(0., .025 - clearance) + .32 * abs(height[2] - BALL_RADIUS)
            + max(0., BALL_RADIUS - height[1]) + max(0., height[3] - BALL_RADIUS)
            + max(0., abs(lateral[2]) - TABLE_HALF_WIDTH + .065)
            + 2. * max(0., .025 - closing))
    return float(cost), float(clearance), bounce


def _incoming_path(estimate, restitution, horizon):
    position, velocity = estimate
    states = [np.r_[position, velocity]]
    for _ in range(horizon):
        position, velocity, _ = _prediction(position, velocity, 1., restitution)
        states.append(np.r_[position, velocity])
    return np.asarray(states)


def _crossing_forecast(ball_path, blade_path, strike, restitution, correction):
    pitch = blade_path[:, 3]
    normal = np.column_stack((np.zeros(len(pitch)), np.cos(pitch), np.sin(pitch)))
    relative = ball_path[:, :3] - blade_path[:, :3]
    gap = np.sum(relative * normal, axis=1) - FACE_DISTANCE
    crossing = np.flatnonzero((gap[:-1] > 0.) & (gap[1:] <= 0.))
    if len(crossing):
        i = crossing[0]
        fraction = gap[i] / (gap[i] - gap[i + 1])
        distance = 0.
    else:
        # Penalize a solution that never reaches the face in the forecast.
        i = int(np.clip(round(strike.time), 1, len(gap) - 1)) - 1
        fraction = 1.
        distance = abs(gap[i + 1]) + .10
    state = ball_path[i] * (1. - fraction) + ball_path[i + 1] * fraction
    blade = blade_path[i] * (1. - fraction) + blade_path[i + 1] * fraction
    motion = blade_path[i + 1, :3] - blade_path[i, :3]
    delta = state[:3] - blade[:3]
    tangent = np.array([0., -sin(blade[3]), cos(blade[3])])
    miss = np.hypot(delta[0], delta @ tangent)
    flight, clearance, bounce = _flight_cost(state[:3], state[3:], motion, blade[3],
                                             restitution, correction)
    cost = (flight + 1.6 * miss + 3. * max(0., miss - .045) + 2.8 * distance
            + .20 * abs(state[1] - strike.point[1]) + .012 * abs(i + fraction - strike.time))
    return Forecast(float(cost), float(i + fraction), tuple(state[:3]), tuple(motion),
                    float(blade[3]), float(miss), clearance, bounce)


class BladeForecast:
    """Approximate the controller's outstanding and next three rate commands."""

    def __init__(self, servo):
        fit = servo.fit
        self.fit = fit
        self.scale = servo.scale[:, None] * servo.weights
        self.requested = servo.requested
        self.correction = servo.correction
        self.times = np.arange(1, servo.horizon + 1)
        depth = fit.kernels.shape[1]
        future = [
            servo.requested + np.sum(servo.commands[:depth - step] * (fit.kernels[:, step:].T - 1.), axis=0)
            for step in range(servo.horizon + 1)
        ]
        free = np.stack([*reversed(servo.nominal_history[1:]), *future])
        delay = int(np.clip(round(servo.observer.delay), 0, len(servo.nominal_history) - 1))
        self.states = free[len(servo.nominal_history) - 1 + self.times - delay]
        self.free_pose = fit.pose(self.states) + self.correction
        age = self.times[:, None] - np.arange(servo.moves) - delay - 1
        self.effect = np.where(age[..., None] >= 0, fit.kernels.T[np.maximum(age, 0)], 0.)
        self.home_pose = np.r_[servo.pose, servo.pitch]
        design = np.einsum('ta,an,tmn->tamn', self.scale, fit.jacobian(servo.effective), self.effect)
        design = design.reshape(len(self.times) * 4, -1)
        self.solve = np.linalg.solve(design.T @ design + .00055 * np.eye(design.shape[1]), design.T)

    def poses(self, strike, rate):
        targets = np.tile(np.r_[strike.target(strike.time, rate), strike.pitch], (len(self.times), 1))
        targets[:, 1] += (np.clip(rate * strike.time, -.107, .145)
                          - np.clip(rate * (strike.time - self.times), -.107, .145))
        error = (targets - self.free_pose) * self.scale
        plan = (self.solve @ error.ravel()).reshape(-1, self.fit.action_size)
        plan = _clipped_plan(plan, self.requested)
        poses = self.fit.pose(self.states + np.sum(self.effect * plan, axis=1)) + self.correction
        return np.vstack((self.home_pose, poses))


class WindowPlayer(RegistrationPlayer):
    """Change the first incoming stroke following a visually observed own serve."""

    def reset(self):
        super().reset()
        self.far_depth = None
        self.reply_tick = None
        self.depth_offset = None
        self.window_pitch_offset = 0.
        self.window_deadline = None
        self.closed = False
        self.window_choices = []

    def _observe_reply(self):
        if self.serve_plan is None or self.reply_tick is not None:
            return
        tick = self.scene.tick
        ball = self.scene.tracker.ball
        if tick <= self.serve_plan[0] + 3 or ball is None or not ball.observed:
            return
        depth = ball.position[1]
        if self.far_depth is not None and .0 < depth < self.far_depth - .14:
            self.reply_tick = tick
        elif depth > .48:
            self.far_depth = depth if self.far_depth is None else max(self.far_depth, depth)

    def _strike(self, estimate, depth, pitch_offset):
        position, velocity = estimate
        time, point, incoming = self.flight.crossing(position, velocity, depth, self._surface_ratio())
        if not -2.8 < time < self.servo.horizon + 1.5 or not .04 < point[2] < .67:
            return None
        bias, gain, lead = self.prior.numeric[0][self.option]
        point[0] += self.transfer_strength * (
            .12 * (gain - 1.) * (position[0] - self.adapter.ready[0])
            + .12 * (lead - 1.) * (point[0] - position[0]) + .035 * bias)
        point[2] = np.clip(point[2], .080, .61)
        pitch = self.memory.choose_angle(point, incoming, self.stroke_speed, self.impact_restitution)
        if self._wrist_available():
            pitch = float(np.clip(pitch + pitch_offset, -.02, .72))
        else:
            pitch = self.adapter.home[3]
        return Strike(time, point, incoming, pitch)

    def _choose_window(self, estimate, fixed):
        tick = self.scene.tick
        blade = BladeForecast(self.servo)
        ball_path = _incoming_path(estimate, self._surface_ratio(), self.servo.horizon)

        def forecast(strike):
            return _crossing_forecast(ball_path, blade.poses(strike, self.stroke_speed),
                                      strike, self.impact_restitution, self.memory.correction())

        comparison = forecast(fixed)
        proposals = []
        for depth_offset in DEPTH_OFFSETS:
            for pitch_offset in PITCH_OFFSETS:
                if depth_offset == pitch_offset == 0.:
                    continue
                strike = self._strike(estimate, fixed.point[1] + depth_offset, pitch_offset)
                if strike is None or not 2. < strike.time < self.servo.horizon:
                    continue
                predicted = forecast(strike)
                cost = predicted.cost + .10 * abs(depth_offset) + .05 * abs(pitch_offset)
                proposals.append((cost, depth_offset, pitch_offset, strike, predicted))
        if not proposals:
            return None
        cost, depth_offset, pitch_offset, strike, predicted = min(proposals, key=lambda row: row[0])
        if cost + MIN_IMPROVEMENT >= comparison.cost:
            return None
        self.depth_offset = depth_offset
        self.window_pitch_offset = pitch_offset
        self.window_deadline = tick + strike.time
        self.window_choices.append(dict(tick=tick, depth_offset=depth_offset, pitch_offset=pitch_offset,
                                        target_time=float(strike.time), baseline=asdict(comparison),
                                        selected=asdict(predicted)))
        return strike

    def _stroke_goals(self, estimate):
        self._observe_reply()
        goal, lead, pitch, horizon = super()._stroke_goals(estimate)
        tick = self.scene.tick
        if self.window_deadline is not None and (self.release == tick or tick > self.window_deadline + 3.):
            self.closed = True
        if (self.reply_tick is None or self.closed or estimate is None or horizon is None
                or self.servo.drive is not None or estimate[1][1] >= -.025):
            return goal, lead, pitch, horizon
        if self.depth_offset is not None:
            strike = self._strike(estimate, self.plan[2] + self.depth_offset, self.window_pitch_offset)
        elif 4. <= horizon <= 8.:
            fixed = Strike(horizon, self.plan[1:4].copy(), estimate[1], pitch)
            strike = self._choose_window(estimate, fixed)
        else:
            strike = None
        if strike is None:
            return goal, lead, pitch, horizon
        self.servo.intercept = strike.time
        self.prediction = strike.incoming[1:].copy(), strike.pitch, self.stroke_speed
        return (strike.target(strike.time, self.stroke_speed), strike.target(strike.time - 1., self.stroke_speed),
                strike.pitch, strike.time)

    def finish(self, reward):
        return dict(super().finish(reward), reply_window=dict(
            optical_reply=self.reply_tick, choices=self.window_choices))
