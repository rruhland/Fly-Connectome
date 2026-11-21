"""Adapt the camera-driven stroke using observed actuator timing and ball flight."""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import lsq_linear

from .tt5_player import BALL_RADIUS, DRAG, GRAVITY, NET_CLEARANCE, TABLE_END, TABLE_HALF_WIDTH
from .tt6_body_control import CameraBodyPlayer, CameraServo, MOTOR_BUDGET, POSE_WEIGHT
from .tt6_continuous import ContinualPlayer, PredictiveServo
from .tt6_timing import TimingMemory, VisualDelay, advance_stroke


@dataclass(frozen=True)
class Stroke:
    speed: float = .049
    pitch_bias: float = 0.
    latency_lead: float = 1.
    depth_shift: float = 0.
    phase_shift: float = 0.


STROKES = (Stroke(), Stroke(.063, .035, .90), Stroke(.074, .130, .87, .070, .20))


def outbound_flight(trace):
    """Read a departing ball arc from the first visually planned contact."""
    planned = [frame['tick'] + frame['horizon'] for frame in trace
               if frame['horizon'] is not None and 0. <= frame['horizon'] < 4.]
    if not planned:
        return None
    contact = planned[0]
    seen = [(frame['tick'], frame['ball']) for frame in trace if frame['ball'] is not None
            and contact - 4 <= frame['tick'] <= contact + 24]
    for start in range(len(seen) - 3):
        window = seen[start:start + 4]
        times = np.array([tick for tick, _ in window], dtype=float)
        points = np.array([point for _, point in window])
        if not .18 > points[0, 1] > -1.85 or np.max(np.diff(times)) > 5 or times[0] > contact + 16:
            continue
        interval = times - times[0]
        corrected = points.copy()
        corrected[:, 2] += .5 * GRAVITY * interval**2
        model = np.column_stack((np.ones(len(times)), interval))
        position, speed = np.linalg.lstsq(model, corrected, rcond=None)[0]
        error = np.linalg.norm(model @ np.stack((position, speed)) - corrected, axis=1)
        if speed[1] <= .014 or speed[1] > .24 or np.max(error) > .075:
            continue
        net_height = None
        for _ in range(42):
            acceleration = -DRAG * np.linalg.norm(speed) * speed - np.array((0., 0., GRAVITY))
            following = position + speed + .5 * acceleration
            if position[1] < 0. <= following[1]:
                fraction = -position[1] / (following[1] - position[1])
                net_height = float(position[2] + fraction * (following[2] - position[2]))
            if following[2] <= BALL_RADIUS:
                ratio = np.clip((position[2] - BALL_RADIUS) / max(.001, position[2] - following[2]), 0., 1.)
                landing = position + ratio * (following - position)
                return dict(landing=landing[:2].tolist(), net_height=net_height, supported_samples=len(times))
            position = following
            speed += acceleration
    return None


class StrokeMemory:
    """Carry scores for a small range of stroke settings across installations."""

    def __init__(self):
        self.counts = np.zeros(len(STROKES), dtype=int)
        self.totals = np.zeros(len(STROKES))

    @property
    def means(self):
        return self.totals / np.maximum(1, self.counts)

    def choose(self, *, explore):
        if explore and self.counts.min() < 4:
            return int(np.argmin(self.counts))
        confidence = .24 * np.sqrt(np.log(self.counts.sum() + 1) / np.maximum(1, self.counts))
        return int(np.argmax(self.means + (confidence if explore else 0.)))

    def observe(self, choice, flight, reward):
        optical = 0.
        if flight is not None:
            x, y = flight['landing']
            optical = np.clip((y + 1.1) / 1.75, 0., 1.)
            optical -= 1.5 * max(0., abs(x) - TABLE_HALF_WIDTH + .10)
            optical -= 1.5 * max(0., y - TABLE_END + .12)
            height = flight['net_height']
            if height is not None:
                optical -= 3. * max(0., NET_CLEARANCE - height)
        score = .70 * np.clip(optical, 0., 1.) + .30 * (reward + 1.) / 2.
        self.counts[choice] += 1
        self.totals[choice] += score
        return float(score)

    def state(self):
        return dict(counts=self.counts.tolist(), means=self.means.tolist(), selected=self.choose(explore=False))


class LagAwareServo(CameraServo):
    """Use the short motor solver and compare camera sightings with delayed poses."""

    def __init__(self, fit, timing, *, carry=True):
        super().__init__(fit)
        self.observer = VisualDelay(timing, fit.ready, np.array([.015, .035, .018]), learn=True)
        self.nominal = deque([self.effective.copy() for _ in self.observer.belief],
                             maxlen=len(self.observer.belief))
        self.carry = carry
        self.previous_sighting = None
        sensitivity = np.linalg.norm(POSE_WEIGHT[:, None] * fit.jacobian(self.requested), axis=0)
        self.regularization = .009 + .026 * np.maximum(0., 1. - sensitivity / .035)**2

    def advance(self, preceding):
        super().advance(preceding)
        if preceding is not None:
            self.nominal.appendleft(self.effective.copy())
            self.observer.advance(self.fit.pose(self.effective)[:3])

    def observe(self, actor, pitch):
        sighting = np.asarray(actor.position) if actor is not None and actor.observed else None
        self.observer.observe(sighting)
        self.effective = self.observer.belief @ np.asarray(self.nominal)
        predicted = self.fit.pose(self.effective)
        if sighting is not None:
            coherent = self.previous_sighting is not None and np.linalg.norm(sighting - self.previous_sighting) < .20
            delta = sighting - predicted[:3]
            if coherent or np.linalg.norm(delta) < .12:
                self.correction[:3] = .40 * self.correction[:3] + .60 * np.clip(delta, -.36, .36)
        self.previous_sighting = sighting
        if pitch is not None and np.isfinite(pitch) and abs(pitch - predicted[3]) < .34:
            self.correction[3] = .65 * self.correction[3] + .35 * (pitch - predicted[3])
        observed = predicted + self.correction
        self.velocity = .35 * self.velocity + .65 * (observed[:3] - self.pose)
        self.pose, self.pitch = observed[:3], observed[3]

    def command(self, goal, lead, pitch):
        baseline = self.requested + np.sum(self.commands[:-2] * (self.fit.kernels[:, 2:].T - 1.), axis=0)
        target = np.r_[lead + .72 * (lead - goal), pitch]
        error = target - self.fit.pose(baseline) - self.correction
        jacobian = self.fit.jacobian(baseline) * self.fit.kernels[:, 1]
        matrix = np.vstack((POSE_WEIGHT[:, None] * jacobian, np.diag(self.regularization)))
        residual = np.r_[POSE_WEIGHT * error, -.32 * (self.regularization - .009) * baseline]
        lower = np.maximum(-1., -MOTOR_BUDGET - self.requested) - 1e-9
        upper = np.minimum(1., MOTOR_BUDGET - self.requested) + 1e-9
        return np.clip(lsq_linear(matrix, residual, bounds=(lower, upper), method='bvls').x, -1., 1.)


class CoherentPredictiveServo(PredictiveServo):
    """Recover large model drift when consecutive camera sightings agree."""

    def __init__(self, fit, timing, *, learn=True):
        super().__init__(fit, timing, learn=learn)
        self.previous_sighting = None

    def observe(self, actor, pitch):
        last_pose, last_velocity = self.pose.copy(), self.velocity.copy()
        super().observe(actor, pitch)
        sighting = np.asarray(actor.position) if actor is not None and actor.observed else None
        if sighting is not None and self.previous_sighting is not None:
            predicted = self.fit.pose(self.effective)[:3]
            innovation = sighting - predicted
            if np.linalg.norm(innovation) >= .16 and np.linalg.norm(sighting - self.previous_sighting) < .20:
                self.correction[:3] = .40 * self.correction[:3] + .60 * np.clip(innovation, -.36, .36)
                self.pose = predicted + self.correction[:3]
                self.velocity = .35 * last_velocity + .65 * (self.pose - last_pose)
        self.previous_sighting = sighting


class CoherentContinualPlayer(ContinualPlayer):
    """Keep the shared predictive stroke and recover from coherent RGB model drift."""

    def _make_servo(self):
        return CoherentPredictiveServo(self.adapter, self.timing_memory, learn=self.adjust)


class RotaryPlayer(CameraBodyPlayer):
    """Change the stroke through its public RGB, motor requests, and point rewards."""

    def __init__(self, adapter, prior, tracker_factory, *, stroke=Stroke(), timing=None, use_delay=True,
                 memory=None, explore=False):
        self.stroke_settings = stroke
        self.timing = TimingMemory() if timing is None else timing
        self.use_delay = use_delay
        self.stroke_memory, self.explore = memory, explore
        super().__init__(adapter, prior, tracker_factory, retain_trace=True, stroke_speed=stroke.speed)

    def _make_servo(self):
        if self.use_delay:
            return LagAwareServo(self.adapter, self.timing)
        return super()._make_servo()

    def reset(self):
        super().reset()
        if self.stroke_memory is not None:
            self.stroke_choice = self.stroke_memory.choose(explore=self.explore)
            self.stroke_settings = STROKES[self.stroke_choice]
        self.stroke_speed = self.stroke_settings.speed

    def _stroke_goals(self, estimate):
        goal, lead, pitch, horizon = super()._stroke_goals(estimate)
        if horizon is None:
            return goal, lead, pitch, horizon
        if self.use_delay:
            delay = self.stroke_settings.latency_lead * self.servo.observer.delay
            goal, lead = advance_stroke(goal, lead, horizon, self.stroke_speed,
                                        delay + self.stroke_settings.phase_shift)
        tilted = float(np.clip(pitch + self.stroke_settings.pitch_bias, -.04, .82))
        center_offset = .027 * np.array((0., math.cos(pitch) - math.cos(tilted),
                                        math.sin(pitch) - math.sin(tilted)))
        return goal + center_offset, lead + center_offset, tilted, horizon

    def _reachable_plane(self, plane, ball):
        return plane + self.stroke_settings.depth_shift

    def finish(self, reward):
        outcome = super().finish(reward)
        flight = outbound_flight(self.trace)
        outcome['stroke_observation'] = flight
        if self.stroke_memory is not None:
            choice = self.stroke_choice
            outcome['stroke_choice'] = choice
            if self.explore:
                outcome['stroke_score'] = self.stroke_memory.observe(choice, flight, reward)
        if self.use_delay:
            observer = self.servo.observer
            observer.finish(carry=self.servo.carry)
            outcome['timing'] = dict(estimated=observer.delay, informative=observer.informative)
        return outcome
