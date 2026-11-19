"""Learn contact offsets from terminal scores over a visual interception policy."""

from dataclasses import dataclass
import math

import numpy as np


# Candidate paddle-center shifts in normalized image coordinates. These define
# the action repertoire. Their values are initially unknown to the learner.
OFFSETS = (0., .079, -.079, .091, -.091, .103, -.103)
CONTEXT_CENTERS = np.array([.04, .22, .41])
CONTEXT_WIDTH = .11


def context_weights(intercept):
    """Share evidence across neighboring arrival heights and mirrored halves."""
    distance = (abs(intercept) - CONTEXT_CENTERS) / CONTEXT_WIDTH
    weights = np.exp(-.5 * np.square(distance))
    return weights / weights.sum()


class ScorePlacement:
    """A contextual bandit with an episode-level score as its only update."""

    def __init__(self, *, totals=None, counts=None, visits=None, updates_enabled=True):
        shape = (len(OFFSETS), len(CONTEXT_CENTERS))
        self.totals = np.zeros(shape) if totals is None else np.array(totals, float)
        self.counts = np.zeros(shape) if counts is None else np.array(counts, float)
        self.visits = np.zeros(len(OFFSETS), int) if visits is None else np.array(visits, int)
        self.updates_enabled = updates_enabled
        if (self.totals.shape != shape or self.counts.shape != shape
                or self.visits.shape != (len(OFFSETS),)
                or not np.isfinite(self.totals).all() or not np.isfinite(self.counts).all()
                or np.any(self.visits < 0) or np.any(self.counts < 0)
                or np.any(abs(self.totals) > self.counts + 1e-9)
                or not np.allclose(self.counts.sum(axis=1), self.visits)):
            raise ValueError('invalid score-placement state')

    def to_state(self):
        return dict(totals=self.totals.tolist(), counts=self.counts.tolist(),
                    visits=self.visits.tolist())

    def frozen_branch(self):
        return ScorePlacement(**self.to_state(), updates_enabled=False)

    def choose(self, intercept, legal, rng, exploration):
        weights = context_weights(intercept)
        visits = self.counts[legal] @ weights
        returns = self.totals[legal] @ weights
        # Near an unvisited context, unwind towards a conservative center tie.
        values = returns / (visits + 1.25)
        if exploration and rng.random() < exploration:
            # Prefer under-sampled actions in the current context on exploration.
            novelty = 1. / np.sqrt(visits + 1.)
            return int(rng.choice(legal, p=novelty / novelty.sum()))
        return int(legal[np.argmax(values)])

    def observe_score(self, arm, intercept, score):
        if not 0 <= arm < len(OFFSETS) or score not in (-1., 0., 1.):
            raise ValueError('legal action and native terminal score required')
        if self.updates_enabled:
            weights = context_weights(intercept)
            self.totals[arm] += weights * score
            self.counts[arm] += weights
            self.visits[arm] += 1

    def episode(self, seed, *, exploration=0.):
        return PlacementEpisode(self, seed, exploration)


@dataclass(frozen=True)
class ContactChoice:
    approach: int
    arm: int
    intercept: float
    offset: float


class PlacementEpisode:
    """Commit once per visually tracked approach, credit the last commitment."""

    def __init__(self, policy, seed, exploration):
        if not 0. <= exploration <= 1.:
            raise ValueError('exploration probability must be in [0, 1]')
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self.exploration = exploration
        self.choices = []
        self.done = False

    def offset(self, approach, intercept, remaining, reachable=(-.4, .4)):
        """Return an offset for the intercept reported by the inherited predictor."""
        if self.done:
            raise ValueError('a completed episode cannot make another choice')
        low, high = reachable
        if (not all(math.isfinite(x) for x in (intercept, remaining, low, high))
                or not -.5 <= intercept <= .5 or low > high or remaining < 0.):
            raise ValueError('a finite visual approach and an ordered reach interval are required')
        choice = self.choices[-1] if self.choices else None
        if choice is None or choice.approach != approach:
            if remaining < .23:
                return 0.
            # Positive repertoire entries shift the paddle away from the image
            # center and negative entries shift it towards the center.
            sign = 1. if intercept >= 0. else -1.
            legal = [0]
            legal.extend(index for index, offset in enumerate(OFFSETS[1:], 1)
                         if low <= intercept + sign * offset <= high)
            if len(legal) == 1:
                return 0.
            arm = self.policy.choose(intercept, legal, self.rng, self.exploration)
            choice = ContactChoice(approach, arm, intercept, sign * OFFSETS[arm])
            self.choices.append(choice)
        return choice.offset

    def finish(self, score):
        if self.done or score not in (-1., 0., 1.):
            raise ValueError('one native terminal score per episode is required')
        self.done = True
        if self.choices:
            choice = self.choices[-1]
            self.policy.observe_score(choice.arm, choice.intercept, score)


class ScorePlacementController:
    """Apply the score policy through the carried timing controller's aim hook."""

    def __init__(self, timing, placement, *, forecast_center=False):
        self.timing = timing
        self.placement = placement
        self.forecast_center = forecast_center
        self.approach = 0
        self.arrival_at = None
        timing.contact_offset = self._offset

    def _offset(self, context):
        if context is None:
            return 0.
        now = self.timing.seconds
        arrival_at = now + context.seconds
        # The projected crossing can jitter or temporarily disappear while
        # the camera resolves the ball near a wall. A later crossing well
        # after the previous predicted contact starts a separate choice.
        if (self.arrival_at is not None and arrival_at - self.arrival_at > .85
                and now > self.arrival_at - .20):
            self.approach += 1
        self.arrival_at = arrival_at
        if context.seconds > 1.10:
            return 0.
        action_time = self.timing.delay_commands * self.timing.camera_seconds
        travel = .64 * max(0., context.seconds - action_time - .12)
        reachable = (max(context.lower, context.paddle - travel),
                     min(context.upper, context.paddle + travel))
        if reachable[0] > reachable[1]:
            return 0.
        offset = self.placement.offset(self.approach, context.intercept,
                                       context.seconds, reachable)
        if not offset and self.forecast_center and self.placement.choices:
            if self.placement.choices[-1].approach == self.approach:
                # Activate the same forecast and braking path with a negligible
                # shift so the control isolates the effect of shot placement.
                offset = 1e-9
        if not offset:
            return 0.
        target = context.intercept + offset
        target = max(context.lower, min(context.upper, target))
        return target - context.intercept

    def choose(self, entities, baseline_action):
        return self.timing.choose(entities, baseline_action)

    def observe(self, **transition):
        self.timing.observe(**transition)
        if transition['terminated'] or transition['truncated']:
            self.placement.finish(transition['reward'])
