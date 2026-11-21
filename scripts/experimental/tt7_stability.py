"""Keep a practiced stroke baseline while correcting from current optical impact."""

from dataclasses import dataclass, field

import numpy as np

from .tt6_continuous import ContinualPlayer


SPEEDS = np.linspace(.049, .101, 9)
SPEED_BANDWIDTH = .009


@dataclass
class StrokeMemory:
    """Smooth the observed public outcomes and regularize where practice is sparse."""

    starting_speed: float
    counts: np.ndarray = field(default_factory=lambda: np.zeros(len(SPEEDS)))
    totals: np.ndarray = field(default_factory=lambda: np.zeros(len(SPEEDS)))
    points: int = 0

    def observe(self, speed, reward):
        weights = np.exp(-.5 * np.square((SPEEDS - speed) / SPEED_BANDWIDTH))
        self.counts += weights
        self.totals += weights * reward
        self.points += 1

    def speed(self):
        # The centered cost keeps weak evidence from pushing the stroke to an
        # extreme. A broad, good range has more influence than a single grid point.
        expected = (self.totals - .15 * 8.) / (self.counts + 8.)
        preference = 7. * expected - .5 * np.square((SPEEDS - self.starting_speed) / .017)
        weight = np.exp(preference - preference.max())
        return float(weight @ SPEEDS / weight.sum())

    def state(self):
        return dict(schema='tt7-stroke-memory-v1', starting_speed=self.starting_speed, counts=self.counts.tolist(),
                    totals=self.totals.tolist(), points=self.points)

    @classmethod
    def from_state(cls, state):
        if state['schema'] != 'tt7-stroke-memory-v1':
            raise ValueError('unsupported stroke memory')
        return cls(state['starting_speed'], np.asarray(state['counts'], dtype=float),
                   np.asarray(state['totals'], dtype=float), state['points'])


class StableStrokePlayer(ContinualPlayer):
    """Carry slower public-outcome learning and retain faster optical correction."""

    def __init__(self, adapter, prior, factory, response, timing, *, stroke_memory):
        self.stroke_memory = stroke_memory
        self.local_correction = 0.
        super().__init__(adapter, prior, factory, response, timing, predictive=True)

    def reset(self):
        self.baseline_speed = self.stroke_memory.speed()
        self.response.speed = float(np.clip(self.baseline_speed + self.local_correction, .049, .102))
        super().reset()

    def finish(self, reward):
        details = super().finish(reward)
        impact = details['impact']
        used, proposed = impact['used_speed'], impact['next_speed']
        if details['opportunity']:
            self.stroke_memory.observe(used, reward)
        # Preserve a successful speed longer. Single noisy estimates of the
        # rebound cannot immediately displace the stroke remembered from many points.
        scale = .25 if reward > 0 else .65
        adjustment = np.clip(scale * (proposed - used), -.014, .014)
        self.local_correction = float(np.clip(.70 * self.local_correction + adjustment, -.022, .022))
        details['stability'] = dict(baseline_speed=self.baseline_speed, proposed_speed=proposed,
                                    next_correction=self.local_correction, observations=self.stroke_memory.points)
        return details
