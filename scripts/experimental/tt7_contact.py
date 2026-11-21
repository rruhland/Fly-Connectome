"""Learn when to raise the racket during a receiving stroke from stereo and score."""

from dataclasses import dataclass

import numpy as np

from .tt5_player import BALL_RADIUS, NET_CLEARANCE, TABLE_END, TABLE_HALF_WIDTH, _prediction
from .tt6_body_control import MOTOR_BUDGET
from .tt6_contact import BASELINE_CONTACT, ContactPlayer
from .tt7_flight import outgoing_speed


@dataclass(frozen=True)
class LiftTiming:
    phase: float
    rise: float = .029

    def offset(self, remaining):
        """Sample the same rising stroke at a different time relative to impact."""
        return self.rise * np.clip(self.phase - remaining, -1.9, 1.8)


@dataclass(frozen=True)
class RateTiming:
    lead: int
    frames: int


TIMINGS = (None, LiftTiming(1.0), LiftTiming(0.), LiftTiming(-1.0))
RATE_TIMINGS = (None, RateTiming(3, 3), RateTiming(2, 2), RateTiming(1, 1))
TIMING_FAMILIES = dict(lift=TIMINGS, rate=RATE_TIMINGS)
CUE_PENALTY = np.diag([2., 7., 7., 10., 10.])


class ContactTimingMemory:
    """Regularized local reward estimates indexed by stroke timing and approach cues."""

    def __init__(self):
        self.counts = np.zeros(len(TIMINGS), dtype=int)
        self.statistics = np.repeat(CUE_PENALTY[None], len(TIMINGS), axis=0)
        self.rewards = np.zeros((len(TIMINGS), len(CUE_PENALTY)))

    def choose(self, cues, *, exploration=True):
        if exploration and self.counts.min() == 0:
            return int(np.argmin(self.counts))
        influence = np.linalg.solve(self.statistics, np.broadcast_to(cues, self.rewards.shape)[..., None])[..., 0]
        values = np.sum(influence * self.rewards, axis=1)
        if exploration:
            values += .12 * np.sqrt(np.log1p(self.counts.sum()) * (influence @ cues))
        return int(np.argmax(values))

    def observe(self, choice, cues, score, weight):
        self.counts[choice] += 1
        self.statistics[choice] += weight * np.outer(cues, cues)
        self.rewards[choice] += weight * (score - .40) * cues

    def state(self):
        return dict(counts=self.counts.tolist(), statistics=self.statistics.tolist(), rewards=self.rewards.tolist())

    @classmethod
    def from_state(cls, state):
        memory = cls()
        counts, statistics, rewards = (np.asarray(state[key]) for key in ('counts', 'statistics', 'rewards'))
        for given, expected in zip((counts, statistics, rewards),
                                   (memory.counts, memory.statistics, memory.rewards), strict=True):
            if given.shape != expected.shape or not np.isfinite(given).all():
                raise ValueError('invalid contact timing memory')
        if np.any(counts < 0) or not np.equal(counts, counts.astype(int)).all():
            raise ValueError('invalid contact timing counts')
        if not np.allclose(statistics, statistics.transpose(0, 2, 1)) or np.min(np.linalg.eigvalsh(statistics)) <= 0.:
            raise ValueError('invalid contact timing statistics')
        memory.counts, memory.statistics, memory.rewards = counts.astype(int), statistics.astype(float), rewards.astype(float)
        return memory


def visual_result(trace, expected):
    """Fit the measured outgoing arc and find a tabletop crossing in camera coordinates."""
    fit = outgoing_speed(trace, expected)
    if fit is None:
        return None
    speed, fitted, ticks = fit
    positions = np.asarray([trace[tick]['ball'] for tick in ticks])
    xs = positions[:, 0]
    clock = np.asarray(ticks, dtype=float)
    vx = (clock - clock.mean()) @ xs / np.sum((clock - clock.mean())**2)
    velocity = np.r_[vx, speed]
    position = fitted.copy()
    net_height = None
    crossing = None
    for _ in range(48):
        following, next_velocity, _ = _prediction(position, velocity, 1., 0.)
        if position[1] < 0. <= following[1]:
            weight = -position[1] / (following[1] - position[1])
            net_height = float(position[2] + weight * (following[2] - position[2]))
        if following[2] <= BALL_RADIUS:
            weight = (position[2] - BALL_RADIUS) / max(.001, position[2] - following[2])
            crossing = (position + np.clip(weight, 0., 1.) * (following - position))[:2]
            break
        position, velocity = following, next_velocity
    if crossing is None:
        return None
    x, y = crossing
    quality = np.clip((y + .95) / 1.35, 0., 1.)
    quality -= 1.8 * max(0., abs(x) - TABLE_HALF_WIDTH + .07)
    quality -= 1.3 * max(0., y - TABLE_END + .10)
    if net_height is not None:
        quality -= 1.55 * max(0., NET_CLEARANCE - net_height)
    return dict(landing=crossing.tolist(), net_height=net_height, quality=float(np.clip(quality, 0., 1.)),
                departure_speed=speed.tolist(), sightings=list(ticks))


class ReceivingTiming:
    """Shape an incoming stroke and update its phase from optical flight and point reward."""

    def shape_contact(self, estimate, goals, fixed):
        goal, lead, pitch, horizon = goals
        if horizon is None:
            return goal, lead, pitch, horizon
        tick = self.scene.tick
        attempt = self.receives[-1] if self.receives else None
        if horizon < 6.5 and (attempt is None or tick - attempt['expected'] > 19.):
            position, velocity = estimate
            lag = self.servo.observer.delay
            cues = np.clip(np.array([1., (abs(position[0]) - .16) / .22, (goal[2] - .27) / .14,
                                     (-velocity[1] - .079) / .045, (lag - 1.) / 2.]), -1.5, 1.5)
            choice = fixed if fixed is not None else self.contact_memory.choose(cues, exploration=self.explore_contact)
            attempt = dict(tick=tick, expected=tick + horizon, profile=choice, cues=cues.tolist(),
                           anchor_updates=0, shaped_frames=0)
            self.receives.append(attempt)
        if attempt is None or tick - attempt['expected'] > 6.:
            return goal, lead, pitch, horizon
        if horizon > 0.:
            attempt['expected'] = tick + horizon
            attempt['anchor_updates'] += 1
        shape = self.timings[attempt['profile']]
        if isinstance(shape, RateTiming) and horizon < 3.8 and 'rate_start' not in attempt:
            attempt['rate_start'] = round(tick + horizon) - shape.lead
        if isinstance(shape, LiftTiming) and horizon < 6.5:
            lag = self.servo.observer.delay
            goal = goal + (0., 0., shape.offset(horizon - lag))
            lead = lead + (0., 0., shape.offset(horizon - lag - 1.))
            attempt['shaped_frames'] += 1
        return goal, lead, pitch, horizon

    def score_contacts(self, reward, fixed):
        for index, attempt in enumerate(self.receives):
            visual = visual_result(self.trace, attempt['expected'])
            point_score = (1. + reward) / 2. if index == len(self.receives) - 1 else .5
            if visual is None:
                score, weight = point_score, .50
            else:
                score, weight = .78 * visual['quality'] + .22 * point_score, 1.
            if fixed is None and self.explore_contact:
                self.contact_memory.observe(attempt['profile'], np.asarray(attempt['cues']), score, weight)
            attempt.update(visual=visual, score=float(score), weight=weight)


class ContactTimingPlayer(ReceivingTiming, ContactPlayer):
    """Choose the phase of a rising visual stroke for each receiving opportunity."""

    def __init__(self, adapter, prior, tracker_factory, *, contact_memory=None, fixed=None, explore=True,
                 coast=True, timing=None, family='lift'):
        self.contact_memory = ContactTimingMemory() if contact_memory is None else contact_memory
        self.fixed = fixed
        self.explore_contact = explore
        self.use_coast = coast
        self.family = family
        self.timings = TIMING_FAMILIES[family]
        self.raise_rates = None if family == 'lift' else self._rate_direction(adapter)
        stroke, follow = BASELINE_CONTACT
        super().__init__(adapter, prior, tracker_factory, stroke=stroke, follow=follow, timing=timing)

    @staticmethod
    def _rate_direction(adapter):
        size = adapter.action_size
        weight = np.array([1., 1., 1., .13])
        jacobian = weight[:, None] * adapter.jacobian(np.zeros(size))
        target = .65 * weight * np.array([0., .035, .047, 0.])
        return np.linalg.lstsq(np.vstack((jacobian, .013 * np.eye(size))),
                               np.r_[target, np.zeros(size)], rcond=None)[0]

    def reset(self):
        super().reset()
        self.last_approach = None
        self.receives = []
        self.coast_ticks = []

    def _coast(self, estimate):
        tick = self.scene.tick
        if estimate is not None:
            position, velocity = estimate
            if velocity[1] < -.028:
                self.last_approach = tick, position, velocity
            else:
                self.last_approach = None
            return estimate
        if not self.use_coast or self.last_approach is None or self.plan is None or not self.far_leg:
            return None
        seen, position, velocity = self.last_approach
        if (not 0 < tick - seen <= 4 or tick - self.plan[0] > 2.
                or tick - self.flight.last[0] > 11 or position[1] > -.78):
            return None
        for _ in range(tick - seen):
            position, velocity, _ = _prediction(position, velocity, 1., self._surface_ratio())
        if position[1] < -1.82 or velocity[1] >= -.028:
            return None
        self.coast_ticks.append(tick)
        return position, velocity

    def _stroke_goals(self, estimate):
        estimate = self._coast(estimate)
        return self.shape_contact(estimate, super()._stroke_goals(estimate), self.fixed)

    def update(self, frames, preceding=None):
        action = super().update(frames, preceding)
        if self.raise_rates is None or not self.receives:
            return action
        attempt = self.receives[-1]
        start = attempt.get('rate_start')
        if start is None:
            return action
        shape = self.timings[attempt['profile']]
        if not start <= self.scene.tick < start + shape.frames:
            return action
        attempt['shaped_frames'] += 1
        lo = np.maximum(-1., -MOTOR_BUDGET - self.servo.requested)
        hi = np.minimum(1., MOTOR_BUDGET - self.servo.requested)
        return np.clip(action + self.raise_rates, lo, hi)

    def finish(self, reward):
        result = super().finish(reward)
        self.score_contacts(reward, self.fixed)
        result.update(contact_timing=self.receives, rescued_approach_frames=self.coast_ticks)
        return result
