"""Learn brick-directed rebounds from public ball tracks and score."""

from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np

from breakout_brick_placement import BrickPlacement, OFFSETS, PIXEL, canonical_camera, target_coverage


SCHEMA = 'visual-rebound-brick-placement-v1'
GAINS = np.linspace(-14., 14., 57)
ANGLE_LIMIT = .97
WINDOW = 9


def centered_forecast(context):
    """Enable the carried arrival forecast with effectively zero shot offset."""
    return 1e-9 if context is not None else 0.


@dataclass(frozen=True)
class Rebound:
    seconds: float
    contact_offset: float
    angle: float


def _line(points, at):
    time = points[:, 0] - at
    centered = time - time.mean()
    position = points[:, 1:3]
    velocity = (centered @ position) / (centered @ centered)
    intercept = position.mean(axis=0) - time.mean() * velocity
    error = np.max(abs(position - intercept - time[:, None] * velocity))
    return intercept, velocity, error


class VisualRebounds:
    """Fit incoming and outgoing rays near the visible paddle rail."""

    def __init__(self):
        self.samples = deque(maxlen=30)

    def observe(self, seconds, entities, pair):
        if pair is None:
            return None
        present = {entity['id']: entity['normalized_position'] for entity in entities if entity['observed']}
        if any(identity not in present for identity in pair):
            return None
        paddle, ball = (present[identity] for identity in pair)
        self.samples.append((seconds, *ball, *paddle))
        if len(self.samples) < 2 * WINDOW or not 0. < ball[1] - paddle[1] < .18:
            return None
        samples = np.asarray(self.samples)
        exit_samples = samples[-WINDOW:]
        if not .09 <= seconds - exit_samples[0, 0] <= .21:
            return None
        outgoing, exit_velocity, exit_error = _line(exit_samples, seconds)
        if not .23 <= exit_velocity[1] <= .74 or exit_error > PIXEL:
            return None
        for end in range(len(samples) - WINDOW, WINDOW - 1, -1):
            start_samples = samples[end - WINDOW:end]
            if seconds - start_samples[-1, 0] > .33:
                break
            if not .09 <= start_samples[-1, 0] - start_samples[0, 0] <= .21:
                continue
            incoming, entry_velocity, entry_error = _line(start_samples, seconds)
            if not -.74 <= entry_velocity[1] <= -.23 or entry_error > PIXEL:
                continue
            interval = (incoming[1] - outgoing[1]) / (exit_velocity[1] - entry_velocity[1])
            when = seconds + interval
            if not start_samples[-1, 0] - .035 <= when <= exit_samples[0, 0] + .035:
                continue
            before = incoming + interval * entry_velocity
            after = outgoing + interval * exit_velocity
            position = .5 * (before + after)
            paddle_position = np.array([np.interp(when, samples[:, 0], samples[:, column]) for column in (3, 4)])
            contact = position - paddle_position
            if (abs(before[0] - after[0]) > 2.5 * PIXEL or abs(contact[0]) > .13
                    or not -PIXEL <= contact[1] <= 3.5 * PIXEL):
                continue
            # Retain outgoing evidence and discard the consumed inbound leg.
            while self.samples and self.samples[0][0] <= when:
                self.samples.popleft()
            return Rebound(float(when), float(contact[0]),
                           math.atan2(float(exit_velocity[0]), float(exit_velocity[1])))
        return None


class VisualReboundPolicy:
    """Calibrate rebound direction from motion and ray payoff from score."""

    def __init__(self, *, updates_enabled=True):
        self.updates_enabled = updates_enabled
        self.probabilities = np.full(len(GAINS), 1. / len(GAINS))
        self.execution_bias = 0.
        self.visual_updates = self.updates = self.scored = 0
        self.payoff_mass = np.array([8., 8.])
        self.payoff_hits = np.array([.96, 7.04])

    @classmethod
    def from_state(cls, state, *, updates_enabled=True):
        if state.get('schema') != SCHEMA:
            raise ValueError('unknown visual rebound schema')
        policy = cls(updates_enabled=updates_enabled)
        policy.probabilities = np.asarray(state['probabilities'], dtype=float).copy()
        policy.execution_bias = float(state['execution_bias'])
        policy.visual_updates = state['visual_updates']
        policy.updates = state['score_updates']
        policy.scored = state['scored']
        policy.payoff_mass = np.asarray(state['payoff_mass'], dtype=float).copy()
        policy.payoff_hits = np.asarray(state['payoff_hits'], dtype=float).copy()
        if (policy.probabilities.shape != GAINS.shape or not np.all(policy.probabilities > 0.)
                or not np.isclose(policy.probabilities.sum(), 1.) or not abs(policy.execution_bias) <= .04
                or any(type(count) is not int or count < 0
                       for count in (policy.visual_updates, policy.updates, policy.scored))
                or policy.scored > policy.updates
                or policy.payoff_mass.shape != (2,) or policy.payoff_hits.shape != (2,)
                or not np.all(np.isfinite(policy.payoff_mass)) or not np.all(policy.payoff_mass > 0.)
                or not np.all((policy.payoff_hits >= 0.) & (policy.payoff_hits <= policy.payoff_mass))):
            raise ValueError('invalid visual rebound state')
        return policy

    def to_state(self):
        return dict(schema=SCHEMA, probabilities=self.probabilities.tolist(), execution_bias=self.execution_bias,
                    visual_updates=self.visual_updates, score_updates=self.updates, scored=self.scored,
                    payoff_mass=self.payoff_mass.tolist(), payoff_hits=self.payoff_hits.tolist())

    def choose(self, intercept, paddle_cross, targets, eligible, rng):
        offsets = OFFSETS[eligible]
        angles = np.clip((self.execution_bias - offsets[:, None]) * GAINS, -ANGLE_LIMIT, ANGLE_LIMIT)
        rays = target_coverage(intercept, paddle_cross, targets, np.tan(angles))
        expected = rays @ self.probabilities
        evidence = expected
        if self.updates_enabled and rng.random() < .22 / (1. + self.visual_updates / 40.):
            evidence = rays[:, rng.choice(len(GAINS), p=self.probabilities)]
        low, high = self.payoff_hits / self.payoff_mass
        values = low + max(.10, high - low) * evidence - .8 * abs(offsets)
        index = int(np.argmax(values))
        return int(eligible[index]), float(expected[index]), float(values[index])

    def observe_rebound(self, rebound, requested_offset=None):
        if not self.updates_enabled:
            return
        angles = np.clip(rebound.contact_offset * GAINS, -ANGLE_LIMIT, ANGLE_LIMIT)
        error = (angles - rebound.angle) / .16
        posterior = self.probabilities * np.exp(-.5 * error**2)
        self.probabilities = .997 * posterior / posterior.sum() + .003 / len(GAINS)
        if requested_offset is not None:
            error = rebound.contact_offset + requested_offset
            self.execution_bias = float(.88 * self.execution_bias + .12 * np.clip(error, -.04, .04))
        self.visual_updates += 1

    def observe_score(self, coverage, score):
        if not self.updates_enabled:
            return
        weights = np.array([1. - coverage, coverage])
        self.payoff_mass += weights
        self.payoff_hits += (score > 0) * weights
        self.updates += 1
        self.scored += int(score > 0)


class VisualReboundPlacement(BrickPlacement):
    """Reuse the inherited reach and camera aiming, with visually verified flights."""

    def __init__(self, timing, policy, seed):
        super().__init__(timing, policy, seed)
        self.rebounds = VisualRebounds()
        self.observed_rebounds = []
        self.shot_rebound = None
        self.clock_offset = 0.

    def camera(self, frame, detector):
        super().camera(*canonical_camera(frame, detector))
        self.clock_offset = detector.seconds - self.timing.seconds

    def offset(self, context):
        return super().offset(context) or centered_forecast(context)

    def observe(self, seconds, entities, pair, score, *, terminated=False, truncated=False):
        if not math.isfinite(score) or score < 0.:
            raise ValueError('nonnegative public brick score required')
        rebound = self.rebounds.observe(seconds, entities, pair)
        if rebound is not None:
            self.observed_rebounds.append(asdict(rebound))
            offset = None
            if self.shot is not None and abs(
                    rebound.seconds - self.clock_offset - self.shot.predicted_contact_seconds) <= .36:
                self.shot_rebound = rebound
                offset = self.shot.offset
            self.policy.observe_rebound(rebound, offset)
        if self.shot_rebound is not None:
            self.shot.score += score
        if terminated or truncated:
            self._finish(scored_only=truncated)

    def _finish(self, *, scored_only=False):
        if self.shot is None:
            return
        self.shot.credited = self.shot_rebound is not None and (bool(self.shot.score) or not scored_only)
        if self.shot.credited:
            self.policy.observe_score(self.likelihood, self.shot.score)
        record = asdict(self.shot)
        record['visual_rebound'] = None if self.shot_rebound is None else asdict(self.shot_rebound)
        self.shots.append(record)
        self.shot = self.shot_rebound = self.likelihood = None
