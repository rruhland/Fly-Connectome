"""Update table-tennis stroke lead after coherent visual delay evidence."""

from collections import deque

import numpy as np

from .tt5_player_body import BodyServo
from .tt6_timing import TimingRallyPlayer, TimingServo


class GatedTimingServo(TimingServo):
    def __init__(self, fit, memory, *, learn=True):
        super().__init__(fit, memory, learn=learn)
        self.confirmed_delay = self.delay_observer.delay
        self.confirmed_belief = self.delay_observer.belief.copy()
        self.recent_delays = deque(maxlen=4)
        self.confirmations = 0

    @property
    def lookahead(self):
        return .85 * self.confirmed_delay

    def observe(self, actor, pitch):
        observer = self.delay_observer
        previous = observer.informative
        observer.observe(actor.position if actor is not None and actor.observed else None)
        if observer.informative > previous:
            center = observer.delay
            nearby = np.abs(observer.hypotheses - center) < 1.35
            if observer.belief[nearby].sum() > .78:
                self.recent_delays.append(center)
            else:
                self.recent_delays.clear()
            if len(self.recent_delays) == self.recent_delays.maxlen and np.ptp(self.recent_delays) < .65:
                target = float(np.median(self.recent_delays))
                change = np.clip(target - self.confirmed_delay, -.48, .48)
                self.confirmed_delay += change
                self.confirmed_belief = observer.belief.copy()
                self.confirmations += 1
        low = int(self.confirmed_delay)
        high = min(low + 1, len(self.responses) - 1)
        fraction = self.confirmed_delay - low
        self.effective = (1. - fraction) * self.responses[low] + fraction * self.responses[high]
        BodyServo.observe(self, actor, pitch)


class GatedTimingRallyPlayer(TimingRallyPlayer):
    """Use the practiced delay until the camera supports a consistent change."""

    def _make_servo(self):
        return GatedTimingServo(self.adapter, self.timing_memory, learn=self.timing_learn)

    def finish(self, reward):
        before = self.timing_memory.probabilities.copy()
        outcome = super().finish(reward)
        if self.timing_learn and self.timing_carry:
            confirmed = self.servo.confirmed_belief if self.servo.confirmations else before
            self.timing_memory.probabilities = .94 * confirmed + .06 / len(confirmed)
            outcome['timing']['remembered'] = self.timing_memory.state()
        outcome['timing']['confirmed_delay'] = self.servo.confirmed_delay
        outcome['timing']['confirmations'] = self.servo.confirmations
        return outcome
