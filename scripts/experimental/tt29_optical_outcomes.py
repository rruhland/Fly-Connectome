"""Correct predicted shot placement using signed errors observed in stereo RGB."""

from collections import deque

import numpy as np
from scipy.optimize import brentq

from .tt29_vision import BOUNCE_Z, VERTICAL, flight_terms


NET_HEIGHT = .9325
CONTEXT_SCALE = np.array([.13, .10, .90, 1.05, 1.05, .075])


def stereo_position(eye):
    """Intersect rays from the latest synchronized exposure without a flight prior."""
    if len(eye.last_frame_views) != 2 or len(eye.rays) < 2:
        return None
    left, right = eye.rays[-2], eye.rays[-1]
    if left.time != eye.last_capture or right.time != left.time:
        return None
    u, v = left.direction, right.direction
    mutual = u @ v
    if abs(mutual) >= .97:
        return None
    offset = right.eye - left.eye
    s, t = offset @ u, offset @ v
    a = left.eye + (s - mutual * t) / (1 - mutual**2) * u
    b = right.eye + (mutual * s - t) / (1 - mutual**2) * v
    if np.linalg.norm(a - b) > .018:
        return None
    return (a + b) / 2


class OutgoingFlight:
    """Fit free flight from observed post-stroke samples up to the first subsequent impact."""

    def __init__(self, strike, speed):
        self.strike = strike
        self.drag = .133 * np.clip(speed, 2.5, 9.)
        self.points = []
        self.segment = []
        self.fit = None
        self.done = False

    def observe(self, time, position):
        if time > self.strike + .74:
            self.done = True
            return
        if position is None or time < self.strike - .12:
            return
        if not self.segment:
            self.points.append((time, position))
            if len(self.points) < 3:
                return
            tail = self.points[-3:]
            if tail[0][1][1] < -.48 and all(
                (b[1][1] - a[1][1]) > 1.25 * (b[0] - a[0]) for a, b in zip(tail, tail[1:])
            ):
                self.segment = tail
            elif time > self.strike + .17:
                self.done = True
                return
            else:
                return
        elif self.fit is not None:
            prev_time, prev_pos = self.segment[-1]
            expected = self.position(time)
            error = np.linalg.norm(position - expected)
            if (
                position[1] < prev_pos[1] + 1.1 * (time - prev_time)
                or error > .035 + .065 * (time - prev_time)
                or (expected[2] < BOUNCE_Z - .012 and position[2] > expected[2] + .008)
            ):
                self.done = True
                return
            self.segment.append((time, position))
        else:
            self.segment.append((time, position))
        times = np.array([t - time for t, _ in self.segment])
        path = np.array([p for _, p in self.segment])
        travel, fall = flight_terms(times, self.drag)
        design = np.column_stack((np.ones_like(travel), travel))
        adjusted = path - fall[:, None] * VERTICAL
        fit, *_ = np.linalg.lstsq(design, adjusted, rcond=None)
        error = np.sqrt(np.mean(np.sum((design @ fit - adjusted) ** 2, axis=1)))
        if error < .018:
            self.fit = time, *fit

    def position(self, time):
        anchor, position, velocity = self.fit
        travel, fall = flight_terms(time - anchor, self.drag)
        return position + travel * velocity + fall * VERTICAL

    def outcome(self):
        if self.fit is None or len(self.segment) < 4:
            return None
        first, last = self.segment[0][0], self.segment[-1][0]
        if last - first < .07 or self.position(first)[2] < BOUNCE_Z:
            return None
        end = first + .85
        if self.position(end)[2] >= BOUNCE_Z:
            return None
        touchdown = brentq(lambda t: self.position(t)[2] - BOUNCE_Z, first, end)
        landing = self.position(touchdown)
        if touchdown - last > .23 or self.position(end)[1] <= 0:
            return None
        crossing = brentq(lambda t: self.position(t)[1], first, end)
        clearance = self.position(crossing)[2] - NET_HEIGHT
        return np.r_[landing[:2], clearance]


class ShotResiduals:
    """Keep a rolling local average of measured placement errors and their magnitude."""

    def __init__(self, checkpoint=None):
        self.samples = deque(maxlen=64)
        if checkpoint is not None:
            if checkpoint["schema"] != "tt29-optical-outcomes-1":
                raise ValueError("Unexpected optical outcome checkpoint schema")
            for entry in checkpoint["samples"]:
                features = np.asarray(entry["features"], dtype=float)
                error = np.asarray(entry["error_m"], dtype=float)
                if features.shape != (len(CONTEXT_SCALE),) or error.shape != (3,):
                    raise ValueError("Unexpected optical outcome checkpoint dimensions")
                if not np.isfinite(features).all() or not np.isfinite(error).all():
                    raise ValueError("Optical outcome checkpoint values must be finite")
                self.add(features, error)

    def checkpoint(self):
        return dict(
            schema="tt29-optical-outcomes-1",
            samples=[dict(features=f.tolist(), error_m=e.tolist()) for f, e in self.samples],
        )

    def add(self, features, residual):
        self.samples.append((features.copy(), np.clip(residual, [-.55, -1.10, -.30], [.55, 1.10, .30])))

    @staticmethod
    def cost(shots):
        x, y, net = shots.T
        return (
            6 * np.maximum(0., abs(x) - .57)
            + 5 * np.maximum(0., .34 - y)
            + 6 * np.maximum(0., y - 1.15)
            + 18 * np.maximum(0., .09 - net)
        )

    def predict(self, features):
        """Return the signed correction and half the local error scale, in meters."""
        if len(self.samples) < 2:
            return np.zeros((len(features), 3)), np.zeros((len(features), 3))
        saved, residuals = map(np.asarray, zip(*self.samples))
        distance = np.sum((features[:, None, :] - saved) ** 2, axis=2)
        weights = np.exp(-.5 * distance)
        total = weights.sum(axis=1)
        # The nominal prediction acts as a prior, so unfamiliar inputs get a small correction.
        correction = (weights @ residuals) / (2. + total[:, None])
        error = np.sqrt((weights @ (residuals**2)) / (3. + total[:, None]))
        return correction, .50 * error

    def penalty(self, features, shots):
        correction, spread = self.predict(features)
        mean = shots + correction
        return np.maximum(0., .5 * (self.cost(mean - spread) + self.cost(mean + spread)) - self.cost(shots))


class OpticalOutcomes:
    """Bind the executed rollout to its RGB flight and apply prior measured errors."""

    def __init__(self, planner, memory, *, learn, freeze=False):
        self.planner = planner
        self.solver = planner.predict
        self.memory = memory
        self.learn = learn
        self.freeze = freeze
        self.batches = []
        self.last_tape = None
        self.pending = None
        self.log = []
        self.changed = 0

    def trajectories(self, goals, target, cutoff):
        rates, scores = self.solver.trajectories(goals, target, cutoff)
        valid = np.flatnonzero(np.isfinite(scores[:, 0]))
        if not len(valid):
            return rates, scores
        arena = self.planner.arena
        path = self.planner.flight.path(
            arena.ball, arena.velocity, arena.spin, 1, arena._received_bounce
        )
        if not len(path):
            return rates, scores
        selection = scores[valid]
        context = np.stack([
            np.interp(selection[:, 1], path[:, 0], path[:, i]) for i in (1, 3, 4, 5, 6)
        ], axis=1)
        context = np.column_stack((context, selection[:, 5])) / CONTEXT_SCALE
        planned = selection[:, 3:6]
        penalty = self.memory.penalty(context, planned)
        self.batches.append((rates, valid, selection, context, penalty, float(arena.data.time)))
        if self.learn:
            scores = scores.copy()
            new_cost = selection[:, 0] + penalty
            self.changed += int(new_cost.argmin() != selection[:, 0].argmin())
            scores[valid, 0] = new_cost
        return rates, scores

    def command(self, tape):
        if tape is not self.last_tape:
            self.last_tape = tape
            if tape is not None:
                self.finish(None)
                for rates, valid, scores, context, penalty, model_time in reversed(self.batches):
                    if not np.shares_memory(rates, tape):
                        continue
                    row = (tape.ctypes.data - rates.ctypes.data) // rates.strides[0]
                    index = np.searchsorted(valid, row)
                    if index < len(valid) and valid[index] == row:
                        score = scores[index]
                        strike = model_time + score[1]
                        flight = OutgoingFlight(strike, score[11])
                        self.pending = flight, context[index], score[3:6], float(penalty[index])
                    break
        self.batches.clear()

    def observe(self, eye):
        if self.pending is not None:
            flight = self.pending[0]
            flight.observe(eye.last_capture, stereo_position(eye))
            if flight.done:
                self.finish(None)

    def finish(self, reward):
        if self.pending is None:
            return
        flight, context, planned, penalty = self.pending
        self.pending = None
        actual = flight.outcome()
        residual = None if actual is None else actual - planned
        if residual is not None and not self.freeze:
            self.memory.add(context, residual)
        self.log.append(dict(
            predicted_strike_s=flight.strike,
            predicted=planned.tolist(),
            measured=None if actual is None else actual.tolist(),
            features=context.tolist(),
            samples=len(flight.segment),
            last_position=None if not flight.segment else flight.segment[-1][1].tolist(),
            reward=reward,
            penalty=penalty,
        ))
