"""Observe actuator latency through a game's camera, and lead moving strokes."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .tt5_player import RallyPlayer
from .tt5_player_body import BodyServo, centers


@dataclass(frozen=True)
class Disturbance:
    """An evaluator's command-delay and noise schedule, in control frames."""

    delays: tuple[tuple[int, int], ...] = ((0, 0),)
    noise: tuple[tuple[int, float], ...] = ((0, 0.),)

    def __post_init__(self):
        for entries in (self.delays, self.noise):
            ticks = [tick for tick, _ in entries]
            if not ticks or ticks[0] != 0 or any(type(tick) is not int for tick in ticks) or ticks != sorted(set(ticks)):
                raise ValueError('a disturbance must start at frame zero and have ordered switches')
        if any(type(delay) is not int or delay < 0 for _, delay in self.delays):
            raise ValueError('expected nonnegative integer actuator delays')
        if any(not np.isfinite(sigma) or sigma < 0. for _, sigma in self.noise):
            raise ValueError('expected finite nonnegative command noise')


class DelayedWiring:
    """Apply a private disturbance to commanded rates during ordinary play."""

    def __init__(self, wiring, disturbance=Disturbance(), *, seed=0):
        self.wiring = wiring
        self.arena = wiring.arena
        self.disturbance = disturbance
        self.seed = seed
        self.action_size = getattr(self.arena, 'action_size', 4)

    def reset(self, **kwargs):
        self.tick = 0
        self.applied_index = -1
        self.pending = deque()
        self.remainder = np.zeros(self.action_size)
        self.delay_index = self.noise_index = 0
        self.applied = []
        self.rng = np.random.default_rng(np.random.SeedSequence([self.seed, kwargs.get('seed') or 0]))
        return self.wiring.reset(**kwargs)

    def step(self, action, *, render=True):
        requested = np.asarray(action, dtype=float)
        if requested.shape != (self.action_size,) or not np.isfinite(requested).all():
            raise ValueError(f'expected {self.action_size} finite commanded rates')
        delays, noise = self.disturbance.delays, self.disturbance.noise
        if self.delay_index + 1 < len(delays) and self.tick == delays[self.delay_index + 1][0]:
            self.delay_index += 1
        if self.noise_index + 1 < len(noise) and self.tick == noise[self.noise_index + 1][0]:
            self.noise_index += 1
        delay, sigma = delays[self.delay_index][1], noise[self.noise_index][1]
        self.pending.append(np.clip(requested, -1., 1.))
        # When transport speeds up, release queued rates together. If it slows,
        # hold the current target until a new command has arrived.
        target_index = self.tick - delay
        requested_rate = self.remainder.copy()
        while self.applied_index < target_index:
            requested_rate += self.pending.popleft()
            self.applied_index += 1
        command = np.clip(requested_rate, -1., 1.)
        self.remainder = requested_rate - command
        if sigma:
            command += self.rng.normal(0., sigma, self.action_size)
        command = np.clip(command, -1., 1.)
        self.applied.append(command.tolist())
        self.tick += 1
        return self.wiring.step(command, render=render)

    @property
    def events(self):
        return self.wiring.events

    def point_result(self):
        return self.wiring.point_result()


class TimingMemory:
    """Persist the most recent visually supported latency belief between points."""

    def __init__(self, probabilities=None, *, max_delay=5):
        if probabilities is None:
            if type(max_delay) is not int or max_delay < 0:
                raise ValueError('expected a nonnegative integer delay limit')
            probabilities = np.exp(-np.arange(max_delay + 1, dtype=float))
        values = np.asarray(probabilities, dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0.) or not values.sum():
            raise ValueError('invalid actuator latency probabilities')
        self.probabilities = values / values.sum()

    def state(self):
        return self.probabilities.tolist()


class VisualDelay:
    """Match camera sightings to recent causal predictions of the controlled object."""

    def __init__(self, memory, initial, noise, *, learn=True):
        self.memory = memory
        self.belief = memory.probabilities.copy()
        self.hypotheses = np.arange(len(self.belief))
        initial = np.asarray(initial, dtype=float).copy()
        self.history = deque([initial] * len(self.belief), maxlen=len(self.belief))
        self.noise = np.broadcast_to(np.asarray(noise, dtype=float), initial.shape)
        if initial.ndim != 1 or not np.isfinite(initial).all() or not np.isfinite(self.noise).all() or np.any(self.noise <= 0.):
            raise ValueError('expected a finite initial observation and positive noise scales')
        self.learn = learn
        self.offset = np.zeros_like(self.noise)
        self.previous = None
        self.informative = 0
        self.trajectory = []
        self.evidence_frames = []

    @property
    def delay(self):
        return float(self.belief @ self.hypotheses)

    def advance(self, nominal):
        self.history.appendleft(np.asarray(nominal, dtype=float).copy())

    def observe(self, sighting):
        predicted = np.asarray(self.history)
        if sighting is None:
            self.previous = None
        elif self.learn:
            measurement = np.asarray(sighting, dtype=float).copy()
            known = np.isfinite(measurement)
            residual = (measurement[known] - predicted[:, known] - self.offset[known]) / self.noise[known]
            distinct = known.any() and np.max(np.ptp(predicted[:, known] / self.noise[known], axis=0)) > .65
            plausible = np.min(np.linalg.norm(residual, axis=1)) < 7.
            if distinct and plausible:
                loss = .65 * np.sum(np.minimum(residual**2, 9.), axis=1)
                if self.previous is not None:
                    previous, old_predictions = self.previous
                    shared = known & np.isfinite(previous)
                    move = (measurement[shared] - previous[shared] - predicted[:, shared]
                            + old_predictions[:, shared]) / (1.4 * self.noise[shared])
                    loss += np.sum(np.minimum(move**2, 9.), axis=1)
                # Permit changes even after a long period of supporting one delay.
                prior = .92 * self.belief + .08 / len(self.belief)
                evidence = np.exp(-.5 * (loss - loss.min()))
                posterior = prior * evidence
                posterior /= posterior.sum()
                # A single ambiguous frame must not advance or retract a stroke.
                movement = abs((posterior - self.belief) @ self.hypotheses)
                weight = min(.32, .28 / max(.28, movement))
                self.belief = (1. - weight) * self.belief + weight * posterior
                self.informative += 1
                self.evidence_frames.append(len(self.trajectory))
            if plausible:
                bias = measurement[known] - self.belief @ predicted[:, known]
                self.offset[known] += .035 * np.clip(bias - self.offset[known], -self.noise[known], self.noise[known])
                self.previous = measurement, predicted
            else:
                self.previous = None
        self.trajectory.append(self.delay)

    def finish(self, *, carry=True):
        if carry and self.learn and self.informative >= 3:
            self.memory.probabilities = .91 * self.belief + .09 / len(self.belief)


class TimingServo(BodyServo):
    """Close the existing signed motor loop around a camera-estimated transport lag."""

    def __init__(self, fit, memory, *, learn=True):
        super().__init__(fit)
        noise = np.array([.013, .021, .014]) * np.clip(fit.error / .018, 1., 2.5)
        self.delay_observer = VisualDelay(memory, fit.ready, noise, learn=learn)
        self.nominal = self.effective.copy()
        self.responses = deque([self.effective.copy()] * len(memory.probabilities), maxlen=len(memory.probabilities))

    @property
    def lookahead(self):
        return self.delay_observer.delay

    def advance(self, preceding):
        self.effective = self.nominal
        super().advance(preceding)
        self.nominal = self.effective
        if preceding is not None:
            self.responses.appendleft(self.nominal)
            self.delay_observer.advance(centers(self.nominal, self.fit.geometry))

    def observe(self, actor, pitch):
        point = actor.position if actor is not None and actor.observed else None
        self.delay_observer.observe(point)
        self.effective = self.delay_observer.belief @ np.asarray(self.responses)
        super().observe(actor, pitch)


def advance_stroke(goal, lead, horizon, stroke_speed, delay):
    """Sample the existing depth trajectory farther ahead in camera frames."""
    # Preserve the predicted intercept and any lateral or height correction.
    time = np.array([horizon, horizon - 1.])
    normal = np.clip(stroke_speed * time, -.107, .145)
    ahead = np.clip(stroke_speed * (time - delay), -.107, .145)
    current_shift, next_shift = normal - ahead
    return goal + np.array([0., current_shift, 0.]), lead + np.array([0., next_shift, 0.])


class TimingRallyMixin:
    """Add a learned actuator lead to any RallyPlayer using the same arm servo."""

    def __init__(self, *args, timing_memory=None, timing_learn=True, timing_carry=True, **kwargs):
        self.timing_memory = TimingMemory() if timing_memory is None else timing_memory
        self.timing_learn = timing_learn
        self.timing_carry = timing_carry
        super().__init__(*args, **kwargs)

    def _make_servo(self):
        return TimingServo(self.adapter, self.timing_memory, learn=self.timing_learn)

    def _stroke_goals(self, estimate):
        goal, lead, pitch, horizon = super()._stroke_goals(estimate)
        if horizon is None:
            return goal, lead, pitch, horizon
        goal, lead = advance_stroke(goal, lead, horizon, self.stroke_speed, self.servo.lookahead)
        return goal, lead, pitch, horizon

    def finish(self, reward):
        outcome = super().finish(reward)
        observer = self.servo.delay_observer
        observer.finish(carry=self.timing_carry)
        outcome['timing'] = dict(belief=observer.belief.tolist(), remembered=self.timing_memory.state(),
                                 informative=observer.informative, trajectory=observer.trajectory,
                                 evidence_frames=observer.evidence_frames)
        return outcome


class TimingRallyPlayer(TimingRallyMixin, RallyPlayer):
    """A first-person table-tennis player with a visual actuator delay observer."""
