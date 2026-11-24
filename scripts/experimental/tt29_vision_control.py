"""Continuous SO-101 joint rates driven by observable ball flight and return outcomes."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .tt4_arena import BALL_RADIUS, NET_HEIGHT, TABLE_HEIGHT
from .tt29_rally_physics import BladeIK
from .tt29_vision import BOUNCE_Z, Flight, VERTICAL, flight_terms


DRAG = .133
RACKET_BOUNCE = .82
CONTACT_OFFSET = BALL_RADIUS + .007
DEFAULT_ACTION = np.zeros(4)
SCALES = np.array([.16, .12, 1.3, 2.2])
EXPLORATION = np.array([
    [0., 0., 0., 0.],
    [-.045, 0., 0., 0.],
    [.045, 0., 0., 0.],
    [-.085, 0., 0., 0.],
    [.085, 0., 0., 0.],
    [0., .40, 0., 0.],
    [0., -.40, 0., 0.],
    [0., 0., .035, 0.],
    [0., 0., -.035, 0.],
    [0., 0., 0., .022],
    [0., 0., 0., -.022],
    [-.045, .40, 0., 0.],
    [.045, .40, 0., 0.],
    [-.045, -.40, 0., 0.],
    [.045, -.40, 0., 0.],
])


def delay_to_y(p, v, damping, y):
    scale = (y - p[1]) / v[1]
    if not (-.06 < scale < 1.) or damping * scale >= .97:
        return np.inf
    return float(-np.log1p(-damping * scale) / damping) if damping else scale


def normal_speed(p, v, landing, pace):
    """Choose an outgoing velocity that crosses the net and lands in a far target."""
    distance = landing[1] - p[1]
    travel = distance / pace
    damping = DRAG * pace
    elapsed = -np.log1p(-travel * damping) / damping
    _, drop = flight_terms(elapsed, damping)
    vz = (BOUNCE_Z - p[2] - drop * VERTICAL[2]) / travel
    half = -p[1] / pace
    _, net_drop = flight_terms(-np.log1p(-damping * half) / damping, damping)
    net_vz = (TABLE_HEIGHT + NET_HEIGHT + BALL_RADIUS + .027 - p[2] - net_drop * VERTICAL[2]) / half
    outgoing = np.array([(landing[0] - p[0]) / travel, pace, max(vz, net_vz)])
    change = outgoing - v
    normal = change / np.linalg.norm(change)
    blade_speed = (outgoing @ normal + RACKET_BOUNCE * (v @ normal)) / (1. + RACKET_BOUNCE)
    return normal, blade_speed


def arc_to_depths(flight, now, depths, material):
    """Predict one bounce from observed flight until a bounce is seen."""
    p, v = flight.advance(now)
    if v[1] >= -.6:
        return []
    sample = Flight(p, v, now, flight.damping, 0., 0., 0, flight.bounced)
    rebound = None
    past_bounce = flight.bounced
    # Low-range wrist views may start just after the bounce is out of sight.
    if not past_bounce and v[2] > .35 and p[1] < -.32 and p[2] > BOUNCE_Z:
        horizon = now - .20
        old = sample.advance(horizon)[0]
        if old[2] < BOUNCE_Z:
            time = brentq(lambda t: sample.advance(t)[0][2] - BOUNCE_Z, horizon, now)
            landing = sample.advance(time)[0]
            past_bounce = abs(landing[0]) < .79 and -1.39 < landing[1] < -.06
    if not past_bounce:
        future = now + .65
        if p[2] >= BOUNCE_Z and sample.advance(future)[0][2] < BOUNCE_Z:
            time = brentq(lambda t: sample.advance(t)[0][2] - BOUNCE_Z, now, future)
            bp, bv = sample.advance(time)
            if abs(bp[0]) < .78 and -1.39 < bp[1] < -.06:
                bv[:2] *= material[0]
                bv[2] *= -material[1]
                rebound = Flight(bp, bv, time, flight.damping, 0., 0., 0, True)
    candidates = []
    for depth in depths:
        use = sample if past_bounce else rebound
        if use is None:
            continue
        eta = delay_to_y(use.position, use.velocity, use.damping, depth)
        total = eta + use.time - now
        if not .005 <= total <= .58 or (not past_bounce and eta < .015):
            continue
        pt, speed = use.advance(now + total)
        if .835 < pt[2] < 1.36 and abs(pt[0]) < .30:
            candidates.append((total, pt, speed))
    return candidates


class OutcomeMemory:
    """Retain how real legal landings depended on observably similar approaches and actions."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.rows = []

    def choose(self, key, *, explore=False):
        if not self.rows:
            return DEFAULT_ACTION.copy()
        keys, options, outcomes = zip(*self.rows, strict=True)
        weight = np.exp(-.5 * np.sum(((np.asarray(keys) - key) / SCALES) ** 2, axis=1))
        evidence = np.bincount(options, weights=weight, minlength=len(EXPLORATION))
        earned = np.bincount(options, weights=weight * outcomes, minlength=len(EXPLORATION))
        if explore:
            scores = self.rng.beta(.35 + earned, 2. + evidence - earned)
        else:
            scores = (.18 + earned) / (1.3 + evidence)
            scores -= np.where(earned > .03, 0., .18)
            scores[0] += .025
        return EXPLORATION[int(np.argmax(scores))].copy()

    def record(self, key, action, reward):
        i = int(np.argmin(np.linalg.norm(EXPLORATION - action, axis=1)))
        self.rows.append((np.asarray(key).copy(), i, float(reward)))


@dataclass
class Stroke:
    time: float
    angles: np.ndarray
    tangent: np.ndarray
    contact: np.ndarray
    orientation: np.ndarray
    direction: np.ndarray
    depth: float
    lead: float
    key: np.ndarray
    action: np.ndarray
    cost: float


class OpticalStroke:
    """Continuous interception using optical state, measured joint state and an issued target."""

    def __init__(self, arena, *, memory=None, explore=False, fixed_action=None):
        self.ik = BladeIK(arena)  # Independent nominal-data FK only.
        self.home = np.asarray(arena._arm_home(1)[0])
        self.dt = float(arena.control_dt)
        actuators = arena._arm_ids[1][2]
        self.rates = arena._joint_increments[actuators] / arena.model.opt.timestep
        self.depths = -arena.base_depth + np.array([.40, .35, .30, .25, .20, .145, .09])
        self.strike_ready = np.array([0., 1.36, -1.12, -1.24, -.80])
        self.memory = OutcomeMemory() if memory is None else memory
        self.explore = explore
        self.fixed_action = fixed_action
        self.stroke = None
        self.last = None
        self.last_plan = -np.inf
        self.last_flight = -np.inf
        self.history = []

    def home_command(self, target):
        return np.clip((self.home - target) / (self.dt * self.rates), -.72, .72)

    def command(self, now, flight, measured, target, material):
        if flight is not None and now - flight.time <= .12 and flight.time > self.last_flight:
            self.last_flight = flight.time
            if flight.velocity[1] > .7:
                self.stroke = None
                self.last_plan = now - .06
            elif self._replan(now):
                result = self._plan(now, flight, measured[:5], target[:5], material)
                self.last_plan = now
                if result is not None:
                    self.stroke = self.last = result
                    self.history.append(dict(time=now, eta=result.time - now, contact=result.contact.tolist(),
                                             normal=result.orientation.tolist(), action=result.action.tolist(),
                                             angles=result.angles.tolist(), tangent=result.tangent.tolist(),
                                             cost=result.cost, expected_direction=result.direction.tolist()))
        if self.stroke is None or now - self.stroke.time > .08:
            return self.home_command(target)
        s = self.stroke
        phase = np.clip(now + self.dt - s.time + s.lead, -.16, .068)
        goals = np.r_[s.angles + s.tangent * phase, self.home[5]]
        return np.clip((goals - target) / (self.dt * self.rates), -1., 1.)

    def _replan(self, now):
        if self.stroke is None:
            return now - self.last_plan >= .062
        eta = self.stroke.time - now
        return eta >= .027 and now - self.last_plan >= (.064 if eta > .15 else .035)

    def _plan(self, now, flight, q, target, material):
        candidates = arc_to_depths(flight, now, self.depths, material)
        best = None
        for eta, p, v in candidates:
            if eta > .55:
                continue
            if self.stroke is not None and self.stroke.time - now < .13 and abs(self.stroke.depth - p[1]) > .058:
                continue
            key = np.array([p[0], p[2], v[1], v[2]])
            action = self.fixed_action
            if action is None:
                action = (self.memory.choose(key, explore=self.explore) if self.stroke is None
                          else self.stroke.action)
            pitch, pace_bias, depth_bias, lead_bias = action
            if self.stroke is None and abs(depth_bias) > .001 and abs(p[1] - (self.depths[1] + depth_bias)) > .13:
                continue
            pace = 4.88 + pace_bias
            normal, blade_speed = normal_speed(p, v, (0., 1.01), pace)
            normal = normal + [0., 0., pitch]
            normal /= np.linalg.norm(normal)
            contact = p - CONTACT_OFFSET * normal
            initial = self.strike_ready.copy()
            initial[0] = np.clip(p[0] / .29, -.75, .75)
            initial[4] -= initial[0]
            if self.stroke is not None:
                initial = self.stroke.angles
            angles, (tip, face, linear, angular) = self.ik.fit(contact, normal, initial)
            discrepancy = np.linalg.norm(tip - contact)
            alignment = np.linalg.norm(face - normal)
            if discrepancy > .065 or alignment > .39:
                continue
            direction = np.linalg.lstsq(np.r_[linear, .095 * angular, .016 * np.eye(5)],
                                        np.r_[normal, np.zeros(8)], rcond=None)[0]
            speed = np.clip(blade_speed, -.10, 1.7)
            tangent = direction * speed
            tangent /= max(1., np.max(abs(tangent) / (.95 * self.rates[:5])))
            achieved = linear @ tangent
            travel = float(np.max(abs(angles - q) / self.rates[:5]))
            margin = eta + .015 - travel
            cost = (25 * discrepancy + 3.5 * alignment + .28 * np.linalg.norm(achieved - normal * speed) +
                    4.4 * max(-margin, 0.) + .45 * max(.06 - margin, 0.) +
                    7.0 * abs(p[1] - (self.depths[1] + depth_bias)))
            if self.stroke is not None:
                cost += 1.1 * abs(p[1] - self.stroke.depth)
            if best is None or cost < best.cost:
                best = Stroke(now + eta, angles, tangent, p, normal, achieved, float(p[1]),
                              .040 + lead_bias, key, action, float(cost))
        return best

    def outcome(self, legal):
        if self.last is not None:
            self.memory.record(self.last.key, self.last.action, legal)
