"""Learn which visual contact placements score on a changing brick board."""

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy import ndimage

from pong_timing_controller import reflect


SCHEMA = 'visual-brick-score-placement-v1'
OFFSETS = np.array([0., .025, -.025, .045, -.045, .065, -.065, .082, -.082])
ANGLE_GAINS = np.repeat(np.array([-14., -10., -6., -2., 2., 6., 10., 14.]), 3)
CONTACT_BIASES = np.tile(np.array([-.024, 0., .024]), 8)
PIXEL = 1 / 64


def canonical_camera(frame, detector):
    """Put the public pixels in the discovered pair's movement coordinates."""
    binding = detector.binding
    pair = detector.pair_ids
    if binding is None or pair is None:
        return None, None
    own = next((item for item in detector.canonical_entities if item['id'] == pair[0]), None)
    if own is None:
        return None, None
    return binding.image(frame), float(own['normalized_position'][1])


def visible_targets(frame, paddle_cross):
    """Read rectangular targets from the canonical public camera image."""
    pixels = np.asarray(frame, dtype=bool)
    if pixels.shape != (64, 64):
        raise ValueError('placement requires a canonical 64 by 64 camera frame')
    labels, _ = ndimage.label(pixels)
    bounds = []
    for identity, (along, cross) in enumerate(ndimage.find_objects(labels), 1):
        w, h = along.stop - along.start, cross.stop - cross.start
        center_along = (along.start + along.stop) * .5 * PIXEL - .5
        center_cross = (cross.start + cross.stop) * .5 * PIXEL - .5
        if not (3 <= w <= 12 and 2 <= h <= 9 and center_cross - paddle_cross >= .24):
            continue
        if np.count_nonzero(labels[along, cross] == identity) < .85 * w * h:
            continue
        bounds.append((center_along, center_cross, w * .5 * PIXEL))
    return np.asarray(bounds, float).reshape(-1, 3)


def predicted_scores(intercept, paddle_cross, targets, offsets):
    """Estimate score chance under each uncalibrated visual rebound hypothesis."""
    if len(targets) == 0:
        return np.full((len(offsets), len(ANGLE_GAINS)), .12)
    along, cross, half_width = targets.T
    angles = np.clip((CONTACT_BIASES - offsets[:, None]) * ANGLE_GAINS, -.97, .97)
    slopes = np.tan(angles)[:, :, None]
    distance = cross - paddle_cross
    outgoing = reflect(intercept + slopes * distance, .5 - PIXEL)
    inbound = reflect(intercept + slopes * (1. - paddle_cross - cross), .5 - PIXEL)
    uncertainty = .022 + .048 * distance

    def coverage(position):
        miss = np.maximum(0., abs(position - along) - half_width - PIXEL)
        return np.exp(-.5 * np.square(miss / uncertainty))

    match = np.maximum(coverage(outgoing), .84 * coverage(inbound))
    return .12 + .76 * match.max(axis=-1)


class BrickShotPolicy:
    """Fit a finite rebound model online with score/no-score evidence."""

    def __init__(self, *, probabilities=None, updates=0, scored=0, updates_enabled=True):
        count = len(ANGLE_GAINS)
        probabilities = (np.full(count, 1. / count) if probabilities is None
                         else np.asarray(probabilities, dtype=float).copy())
        if (probabilities.shape != (count,) or not np.isfinite(probabilities).all()
                or np.any(probabilities <= 0.) or not np.isclose(probabilities.sum(), 1.)
                or type(updates) is not int or type(scored) is not int or not 0 <= scored <= updates):
            raise ValueError('invalid brick-placement state')
        self.probabilities = probabilities
        self.updates = updates
        self.scored = scored
        self.updates_enabled = updates_enabled

    @classmethod
    def from_state(cls, state, *, updates_enabled=True):
        if state.get('schema') != SCHEMA:
            raise ValueError('unknown brick-placement schema')
        return cls(probabilities=state['probabilities'], updates=state['updates'],
                   scored=state['scored'], updates_enabled=updates_enabled)

    def to_state(self):
        return dict(schema=SCHEMA, probabilities=self.probabilities.tolist(),
                    updates=self.updates, scored=self.scored)

    def choose(self, intercept, paddle_cross, targets, eligible, rng):
        offsets = OFFSETS[eligible]
        likelihoods = predicted_scores(intercept, paddle_cross, targets, offsets)
        # A small share of learning shots follow one sampled rebound model.
        # Frozen branches use the probability-weighted estimate every time.
        explore = self.updates_enabled and rng.random() < .22 / (1. + self.updates / 80.)
        if explore:
            model = rng.choice(len(self.probabilities), p=self.probabilities)
            values = likelihoods[:, model]
        else:
            values = likelihoods @ self.probabilities
        values = values - .8 * abs(offsets)
        index = int(np.argmax(values))
        return int(eligible[index]), likelihoods[index], float(values[index])

    def observe_score(self, likelihood, score):
        if not self.updates_enabled:
            return
        evidence = likelihood if score > 0 else 1. - likelihood
        posterior = self.probabilities * np.sqrt(evidence)
        # Keep enough support to adapt if timing changes effective contact.
        self.probabilities = .997 * posterior / posterior.sum() + .003 / len(posterior)
        self.updates += 1
        self.scored += int(score > 0)


@dataclass
class Shot:
    at_seconds: float
    predicted_contact_seconds: float
    intercept: float
    offset: float
    predicted_value: float
    visible_targets: int
    score: float = 0.
    credited: bool = False


class BrickPlacement:
    """Use the carried timing controller's normal contact-offset callback."""

    def __init__(self, timing, policy, seed):
        self.timing = timing
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self.frame = None
        self.paddle_cross = None
        self.arrival_at = None
        self.shot = None
        self.likelihood = None
        self.shots = []
        timing.contact_offset = self.offset

    def camera(self, canonical_frame, paddle_cross):
        """Receive public camera geometry in the role controller's coordinates."""
        self.frame = canonical_frame
        self.paddle_cross = paddle_cross

    def offset(self, context):
        if context is None or context.seconds > 1.10:
            return 0.
        now = self.timing.seconds
        arrival = now + context.seconds
        if (self.arrival_at is not None and arrival - self.arrival_at > .65
                and now > self.arrival_at - .14):
            self._finish()
            self.arrival_at = None
        self.arrival_at = arrival
        if self.shot is None:
            self._choose_shot(context, now, arrival)
        if self.shot is None:
            return 0.
        target = np.clip(context.intercept + self.shot.offset, context.lower, context.upper)
        # Use the same pending-command forecast even for an eligible center shot.
        return float(target - context.intercept) or 1e-9

    def _choose_shot(self, context, now, arrival):
        if self.frame is None or self.paddle_cross is None:
            return
        lag = self.timing.delay_commands * self.timing.camera_seconds
        remaining = context.seconds - lag
        if remaining < .23:
            return
        travel = .64 * (remaining - .12)
        legal = (max(context.lower, context.paddle - travel),
                 min(context.upper, context.paddle + travel))
        destinations = context.intercept + OFFSETS
        eligible = np.flatnonzero((destinations >= legal[0]) & (destinations <= legal[1]))
        if not len(eligible) or not np.any(eligible != 0):
            return
        targets = visible_targets(self.frame, self.paddle_cross)
        if not len(targets):
            return
        arm, self.likelihood, value = self.policy.choose(
            context.intercept, self.paddle_cross, targets, eligible, self.rng)
        self.shot = Shot(now, arrival, context.intercept, float(OFFSETS[arm]), value, len(targets))

    def observe_score(self, score, *, terminated=False, truncated=False):
        """Accumulate public brick reward. Close a flight at the next approach."""
        if not math.isfinite(score) or score < 0.:
            raise ValueError('nonnegative public brick score required')
        if self.shot is not None:
            self.shot.score += score
        if terminated or truncated:
            self._finish(scored_only=True)

    def _finish(self, *, scored_only=False):
        shot = self.shot
        if shot is None:
            return
        # An unexplained zero on termination could be a failed interception
        # or a time cap. A zero between two approaches is a completed flight.
        shot.credited = bool(shot.score or not scored_only)
        if shot.credited:
            self.policy.observe_score(self.likelihood, shot.score)
        self.shots.append(asdict(shot))
        self.shot = self.likelihood = None
