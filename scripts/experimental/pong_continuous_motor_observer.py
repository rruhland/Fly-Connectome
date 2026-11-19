"""Infer independent, continuous action effects from a learned visual motion model."""

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import math

import numpy as np

from pong_relational_persistent_binding import PersistentRelationalController


@dataclass(frozen=True)
class VisualPlant:
    """Two coupled low-pass filters fitted from normal public camera motion."""

    response_seconds: float
    coast_seconds: float
    strength: float
    speed_limit: float
    travel_limit: float
    idle: float

    def __post_init__(self):
        if (not all(math.isfinite(value) for value in asdict(self).values())
                or min(self.response_seconds, self.coast_seconds, self.strength,
                       self.speed_limit, self.travel_limit) <= 0
                or abs(self.idle) >= self.strength):
            raise ValueError('positive visual dynamics and an interior idle gain required')

    @property
    def normal_gains(self):
        return np.array([self.strength, self.idle, -self.strength])

    @lru_cache(maxsize=128)
    def interval(self, elapsed):
        response = self.response_seconds
        coast = self.coast_seconds
        lag_decay = math.exp(-elapsed / response)
        velocity_decay = math.exp(-elapsed / coast)
        velocity_to_position = coast * (1 - velocity_decay)
        if abs(response - coast) < 1e-8:
            lag_to_velocity = elapsed / coast * velocity_decay
            lag_to_position = velocity_to_position - elapsed * velocity_decay
        else:
            rate = response / (response - coast)
            lag_to_velocity = rate * (lag_decay - velocity_decay)
            lag_to_position = rate * (response * (1 - lag_decay) - velocity_to_position)
        transition = np.array([[1., velocity_to_position, lag_to_position],
                               [0., velocity_decay, lag_to_velocity],
                               [0., 0., lag_decay]])
        drive = np.array([elapsed - velocity_to_position - lag_to_position,
                          1 - velocity_decay - lag_to_velocity, 1 - lag_decay])
        return transition, drive


@dataclass
class GainFilter:
    """A Gaussian over position, velocity, filtered drive, and three action gains."""

    plant: VisualPlant
    mean: np.ndarray
    covariance: np.ndarray

    @classmethod
    def start(cls, plant, position, gains, gain_covariance):
        covariance = np.diag([.005 ** 2, .15 ** 2, .28 ** 2, 0., 0., 0.])
        covariance[3:, 3:] = gain_covariance
        return cls(plant, np.r_[position, 0., 0., gains], covariance)

    def fresh_gains(self):
        mean = self.mean.copy()
        mean[3:] = 0.
        covariance = np.zeros_like(self.covariance)
        covariance[:3, :3] = self.covariance[:3, :3]
        covariance[3:, 3:] = np.eye(3) * (1.2 * self.plant.strength) ** 2
        return replace(self, mean=mean, covariance=covariance)

    def observe(self, action, elapsed, position, gain_drift):
        transition = np.eye(6)
        transition[:3, :3], transition[:3, 3 + action] = self.plant.interval(elapsed)
        self.mean = transition @ self.mean
        self.covariance = transition @ self.covariance @ transition.T
        if abs(self.mean[0]) > self.plant.travel_limit:
            outward = np.sign(self.mean[0])
            self.mean[0] = outward * self.plant.travel_limit
            if self.mean[1] * outward > 0:
                self.mean[1] = 0.
        self.mean[1] = np.clip(self.mean[1], -self.plant.speed_limit, self.plant.speed_limit)
        process_std = np.array([.0005, .009, .015, gain_drift, gain_drift, gain_drift])
        self.covariance.flat[::7] += (elapsed / .02) * process_std ** 2
        if position is None:
            return 0.
        residual = position - self.mean[0]
        variance = self.covariance[0, 0] + .005 ** 2
        cross = self.covariance[:, 0].copy()
        self.mean += cross * (residual / variance)
        self.covariance -= np.outer(cross, cross) / variance
        return -.5 * (math.log(variance) + residual ** 2 / variance)


@dataclass
class RestartCandidate:
    observer: GainFilter
    score: float = 0.
    seconds: float = 0.


class ContinuousMotorEffects:
    """Carry gains across games and broaden uncertainty on observed model failure."""

    def __init__(self, plant, *, updates_enabled=True):
        self.plant = plant
        self.updates_enabled = updates_enabled
        self.effects = plant.normal_gains.tolist()
        self.covariance = np.zeros((3, 3)).tolist()
        self.completed_calibrations = 0
        self.contradictions = 0
        self.dither_count = 0
        self.adapted = False

    def request(self, intended):
        return int(np.argsort(-np.asarray(self.effects), kind='stable')[intended])

    @property
    def variances(self):
        return [self.covariance[button][button] for button in range(3)]

    def state(self):
        return dict(schema='public-continuous-motor-effects-v1', plant=asdict(self.plant),
                    effects=self.effects.copy(), covariance=[row.copy() for row in self.covariance],
                    completed_calibrations=self.completed_calibrations,
                    contradictions=self.contradictions, dither_count=self.dither_count,
                    adapted=self.adapted, updates_enabled=self.updates_enabled)

    @classmethod
    def from_state(cls, state):
        if state['schema'] != 'public-continuous-motor-effects-v1':
            raise ValueError('unknown continuous motor effects checkpoint')
        learner = cls(VisualPlant(**state['plant']), updates_enabled=state['updates_enabled'])
        effects, covariance = np.asarray(state['effects'], float), np.asarray(state['covariance'], float)
        if (effects.shape != (3,) or covariance.shape != (3, 3)
                or not np.isfinite(effects).all() or not np.isfinite(covariance).all()
                or not np.allclose(covariance, covariance.T)
                or np.linalg.eigvalsh(covariance)[0] < -1e-10):
            raise ValueError('finite gains and a positive semidefinite covariance required')
        learner.effects, learner.covariance = effects.tolist(), covariance.tolist()
        learner.completed_calibrations = state['completed_calibrations']
        learner.contradictions = state['contradictions']
        learner.dither_count = state['dither_count']
        learner.adapted = state['adapted']
        return learner

    def frozen_branch(self):
        checkpoint = self.state()
        checkpoint['updates_enabled'] = False
        return self.from_state(checkpoint)


class ContinuousMotorController:
    """Preserve inherited intent and seek evidence in the currently useful direction."""

    def __init__(self, policy, learner):
        self.inherited = PersistentRelationalController(policy)
        self.learner = learner
        self.motor = None
        self.axis = policy.original.axis
        self.observer = None
        self.candidates = []
        self.elapsed = 0.
        self.last_restart = -1.
        self.last_position = None
        self.last_action = None
        self.last_intended = None
        self.action_seconds = 0.
        self.pending = None
        self.order_confirmed = False

    def position(self, entities):
        for entity in entities:
            if entity['id'] == self.motor and entity['observed']:
                return float(entity['normalized_position'][self.axis])
        return None

    def choose(self, entities, baseline_action):
        intended, decision = self.inherited.choose(entities, baseline_action)
        motor = decision['bound_ids'][0] if decision['bound_ids'] else self.motor
        if motor != self.motor:
            self.motor = motor
            self.observer = None
            self.candidates.clear()
            self.last_position = None
        position = self.position(entities)
        plant = self.learner.plant
        if self.observer is None and position is not None:
            self.observer = GainFilter.start(plant, position, self.learner.effects,
                                             self.learner.covariance)
            self.last_position = position
        action = self.learner.request(intended)
        if self.learner.adapted and self.learner.updates_enabled and self.observer is not None:
            action = self._choose_continuous(intended)
        if action != self.last_action:
            self.action_seconds = 0.
        self.last_intended, self.last_action = intended, action
        self.pending = action
        return action, dict(intended=intended, action=action, gains=self.learner.effects.copy())

    def _choose_continuous(self, intended):
        gains = self.observer.mean[3:]
        uncertainty = np.sqrt(np.maximum(self.observer.covariance.diagonal()[3:], 0.))
        if intended == 1:
            # Small expected absolute effects keep the paddle near its current point.
            return int(np.argmin(abs(gains) + .15 * uncertainty))
        direction = 1 if intended == 0 else -1
        optimistic = direction * gains + .75 * uncertainty
        action = int(np.argmax(optimistic))
        if (intended == self.last_intended and self.last_action is not None
                and uncertainty[self.last_action] > .28 and 0 < self.action_seconds < .08):
            action = self.last_action
        if action != int(np.argmax(direction * gains)):
            self.learner.dither_count += 1
        return action

    def observe(self, *, own_hits, reward, terminated, truncated, elapsed_seconds, public_entities):
        if self.pending is None or not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            raise ValueError('a chosen request and positive public elapsed time are required')
        action, self.pending = self.pending, None
        self.inherited.observe(own_hits, terminated)
        if self.observer is None or not self.learner.updates_enabled:
            return
        if self.elapsed - self.last_restart >= .075:
            self.candidates.append(RestartCandidate(self.observer.fresh_gains()))
            self.last_restart = self.elapsed
        self.elapsed += elapsed_seconds
        self.action_seconds += elapsed_seconds
        position = self.position(public_entities)
        if (self.last_position is not None and position is not None
                and abs(position - self.last_position) > .045):
            position = None
        self.last_position = position
        baseline_score = self.observer.observe(action, elapsed_seconds, position,
                                               gain_drift=.005 if self.learner.adapted else 0.)
        for candidate in self.candidates:
            candidate.seconds += elapsed_seconds
            candidate.score += candidate.observer.observe(action, elapsed_seconds, position, 0.) - baseline_score
        self.candidates = [candidate for candidate in self.candidates if candidate.seconds < .34]
        if self.candidates:
            best = max(self.candidates, key=lambda candidate: candidate.score)
            if best.score >= (8. if self.learner.adapted else 5.):
                self.observer = best.observer
                self.candidates.clear()
                self.learner.adapted = True
                self.learner.contradictions += 1
                self.order_confirmed = False
        if self.learner.adapted:
            self.learner.effects = self.observer.mean[3:].tolist()
            self.learner.covariance = self.observer.covariance[3:, 3:].tolist()
            self._record_supported_order()

    def _record_supported_order(self):
        if self.order_confirmed:
            return
        covariance = self.observer.covariance[3:, 3:]
        gains = self.observer.mean[3:]
        order = np.argsort(gains)
        for lower, upper in zip(order[:-1], order[1:], strict=True):
            variance = covariance[lower, lower] + covariance[upper, upper] - 2 * covariance[lower, upper]
            if gains[upper] - gains[lower] <= 2 * math.sqrt(max(variance, 0.)):
                return
        self.order_confirmed = True
        self.learner.completed_calibrations += 1
