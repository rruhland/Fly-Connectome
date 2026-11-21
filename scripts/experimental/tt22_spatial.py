"""Carry native Pong score-selected relative motion gains into a TT contact target."""

import numpy as np

from .tt22_rally import RallyPlayer


POPULATION = 6
SOURCE_PRIOR = 8.
MAX_CORRECTION = .060


def broad_hypotheses(seed):
    """Use the original TT3 generation-one signed search, before any source play."""
    rng = np.random.default_rng(seed)
    quantiles = np.stack([(rng.permutation(POPULATION)+rng.random(POPULATION))*2/POPULATION-1
                          for _ in range(3)], axis=1)
    return quantiles*np.array((.11, 1.4, 1.4))


class SpatialChoice:
    """Use the same terminal reward update and exploration rule with each prior."""

    def __init__(self, common, source, mode, ties):
        native = np.asarray(source['coefficients'])
        means = np.asarray(source['totals'], dtype=float) / np.asarray(source['counts'], dtype=float)
        shared = np.asarray(common, dtype=float)
        self.coefficients = shared if mode == 'scratch' else np.concatenate((shared, native))
        self.prior_count = np.ones(len(self.coefficients))
        self.prior_total = np.zeros(len(self.coefficients))
        if mode != 'scratch':
            if mode == 'score_permuted':
                means = np.roll(means, len(means)//2)
            self.prior_count[len(shared):] = SOURCE_PRIOR
            self.prior_total[len(shared):] = SOURCE_PRIOR * means
        self.ties = np.asarray(ties)[:len(self.coefficients)]
        self.count = np.zeros(len(self.coefficients), dtype=int)
        self.total = np.zeros(len(self.coefficients))

    def choose(self, explore=False):
        score = (self.total+self.prior_total)/(self.count+self.prior_count)
        if explore:
            score += .42*np.sqrt(np.log(2 + self.count.sum())/(1+self.count))
        return int(np.argmax(score + 1.e-8*self.ties))

    def update(self, option, reward, returns):
        self.count[option] += 1
        self.total[option] += reward + .35*min(2, returns)

    def state(self):
        pick = self.choose()
        return dict(coefficients=self.coefficients.tolist(), prior_count=self.prior_count.tolist(),
                    prior_total=self.prior_total.tolist(), counts=self.count.tolist(), totals=self.total.tolist(),
                    ties=self.ties.tolist(), greedy=pick, greedy_coefficients=self.coefficients[pick].tolist())


def portable_contact(actor_x, mover_x, forecast_x, coefficients):
    """Carry only portable gains. The 2-D source bias has no signed 3-D axis."""
    _, displacement, motion = coefficients
    return actor_x + displacement*(mover_x-actor_x) + motion*(forecast_x-mover_x)


class SpatialPlayer(RallyPlayer):
    def __init__(self, law, *, direct=False):
        self.direct = direct
        super().__init__(law)

    def reset(self):
        super().reset()
        self.actor = None
        self.actor_seen = -100
        self.frames = 0
        self.applied = 0
        self.neutral_fallbacks = 0
        self.max_correction = 0.
        self.coefficients = (0., 1., 1.)
        self.flat_center = False
        self.geometric_override = False

    def observe_actor(self, tick, position):
        self.frames = tick
        if position is not None:
            self.actor = np.asarray(position)
            self.actor_seen = tick

    def lateral_contact(self, estimate, center):
        x = center[0]
        if self.geometric_override:
            return x
        if self.flat_center:
            return 0.
        visible = self.actor is not None and self.frames - self.actor_seen <= 4
        if not visible:
            if not self.direct:
                return x
            self.neutral_fallbacks += 1
            requested = 0.
        else:
            requested = portable_contact(self.actor[0], estimate[0][0], x, self.coefficients)
        correction = float(requested-x if self.direct else np.clip(requested-x, -MAX_CORRECTION, MAX_CORRECTION))
        self.applied += abs(correction) > 1.e-6
        self.max_correction = max(self.max_correction, abs(correction))
        return float(requested) if self.direct else x+correction
