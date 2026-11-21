"""Search a continuous service stroke with optical flight and point reward."""

from dataclasses import dataclass

import numpy as np

from .tt5_player import BALL_RADIUS, NET_CLEARANCE, TABLE_END, _prediction
from .tt7_flight import FlightPlayer, outgoing_speed
from .tt8_service_drive import ServiceDriveServo
from .tt8_service_player import falling_crossing


@dataclass(frozen=True)
class ServiceStroke:
    height: float = .45
    pitch: float = .19
    forward: float = .132
    lift: float = 0.
    lead: float = 0.
    depth: float = 0.


BOUNDS = np.array([(.345, .545), (-.015, .46), (.095, .185), (-.055, .027), (-.80, 1.75), (-.055, .055)])
EXCURSION = np.array([.060, .125, .035, .034, .95, .038])


def proposals(stroke, rng, generation):
    """Compare an incumbent with two mirrored random directions on the same toss."""
    origin = np.array(tuple(vars(stroke).values()))
    scale = max(.37, .88**generation)
    perturbations = rng.normal(size=(2, len(origin)))
    perturbations /= np.linalg.norm(perturbations, axis=1)[:, None] / np.sqrt(len(origin))
    offsets = np.concatenate([perturbations, -perturbations], axis=0) * EXCURSION * scale
    nearby = np.clip(origin + offsets, BOUNDS[:, 0], BOUNDS[:, 1])
    return [stroke, *(ServiceStroke(*map(float, row)) for row in nearby)]


class SearchServo(ServiceDriveServo):
    lift_rate = 0.

    def _strike_objective(self, normal, rhs, states, effect, jac, targets):
        super()._strike_objective(normal, rhs, states, effect, jac, targets)
        if self.lift_rate == 0. or self.drive is None or not 0. < self.intercept < self.horizon + 1.:
            return
        motion = np.diff(self.fit.pose(states)[:, 2])
        response = np.einsum('n,tmn->tmn', jac[2], np.diff(effect, axis=0))
        times = np.arange(2, self.horizon + 1) - .5
        weight = 2.2 * np.exp(-.5 * ((times - self.intercept) / 1.4)**2)
        rows = (response * weight[:, None, None]).reshape(len(times), -1)
        normal += rows.T @ rows
        rhs += rows.T @ ((self.lift_rate - motion) * weight)


def _project_service(position, speed):
    """Estimate the first bounce and clearance from a local optical launch fit."""
    velocity = np.array([0., *speed])
    first_bounce = clearance = None
    for _ in range(28):
        previous = position
        position, velocity, bounced = _prediction(position, velocity, 1., .87)
        if bounced and first_bounce is None:
            first_bounce = float(position[1])
        if previous[1] < 0. <= position[1]:
            fraction = -previous[1] / (position[1] - previous[1])
            clearance = float(previous[2] + fraction * (position[2] - previous[2]))
            break
        if position[2] < -BALL_RADIUS:
            break
    return first_bounce, clearance


def optical_credit(trace, strike, depth, reward):
    """Score coherent down-table travel, a near-side bounce, and net passage."""
    recent = [(row['tick'], row['ball']) for row in trace if row['ball'] is not None
              and strike <= row['tick'] <= strike + 29]
    depths = [min(a[1][1], b[1][1]) for a, b in zip(recent, recent[1:]) if 0 < b[0] - a[0] <= 4
              and abs(a[1][1] - b[1][1]) < .55]
    advance = max(depths, default=depth) - depth
    bounce = None
    for (ta, a), (tb, b), (tc, c) in zip(recent, recent[1:], recent[2:]):
        if (tb - ta <= 4 and tc - tb <= 4 and a[1] < b[1] < c[1] and b[2] < .12
                and a[2] - b[2] > .019 and c[2] - b[2] > .019):
            bounce = b[1]
            break
    fit = outgoing_speed(trace, strike)
    speed = clearance = None
    if fit is not None:
        speed = float(fit[0][0])
        projected_bounce, clearance = _project_service(fit[1], fit[0])
        if bounce is None:
            bounce = projected_bounce
    score = .42 * float(np.clip(advance / 2.10, 0., 1.))
    score += .22 * float(-TABLE_END - .045 <= bounce < -.12) if bounce is not None else 0.
    if clearance is not None:
        score += .12 * float(np.clip((clearance - NET_CLEARANCE + .06) / .12, 0., 1.))
    if speed is not None:
        score += .075 * float(np.clip(speed / .115, 0., 1.))
    score += .18 * max(0., reward)
    return dict(score=score, travel=float(advance), bounce=None if bounce is None else float(bounce),
                clearance=clearance, speed=speed, reward=reward)


class SearchPlayer(FlightPlayer):
    def __init__(self, *args, stroke, **kwargs):
        self.service_stroke = stroke
        super().__init__(*args, **kwargs)

    def _make_servo(self):
        return SearchServo(self.adapter, self.timing_memory, learn=self.adjust)

    def reset(self):
        super().reset()
        self.toss = []
        self.serve_plan = None

    def _stroke_goals(self, estimate):
        tick = self.scene.tick
        seen = self.scene.tracker.ball
        stroke = self.service_stroke
        self.servo.drive = None
        if (tick <= 18 and seen is not None and seen.observed and seen.position[1] < -.98
                and (self.serve_plan is None or tick < self.serve_plan[0] - 2)):
            self.toss.append((tick, seen.position))
            if (len(self.toss) >= 3 and max(row[1][2] for row in self.toss) > self.toss[0][1][2] + .05
                    and np.ptp([row[1][1] for row in self.toss]) < .11):
                crossing = falling_crossing(self.toss[-9:], stroke.height)
                if crossing is not None and tick + 1 <= crossing[0] <= tick + 22:
                    self.serve_plan = crossing
        if self.serve_plan is None or tick > self.serve_plan[0] + 3:
            return super()._stroke_goals(estimate)

        deadline, xy = self.serve_plan
        horizon = deadline - stroke.lead - tick
        normal = np.array([0., np.cos(stroke.pitch), np.sin(stroke.pitch)])
        centre = np.array([xy[0], xy[1] + stroke.depth, stroke.height]) - .024 * normal

        def position(time):
            y = np.clip(stroke.forward * time, -.107, .185)
            z = np.clip(stroke.lift * time, -.080, .080)
            return centre - np.array([0., y, z])

        self.servo.intercept = horizon
        self.servo.stroke = stroke.forward
        self.servo.drive = stroke.forward
        self.servo.lift_rate = stroke.lift
        return position(horizon), position(horizon - 1.), stroke.pitch, horizon

    def finish(self, reward):
        details = super().finish(reward)
        details['optical_service'] = None
        if self.serve_plan is not None:
            deadline, xy = self.serve_plan
            details['optical_service'] = optical_credit(self.trace, deadline, xy[1], reward)
        return details
