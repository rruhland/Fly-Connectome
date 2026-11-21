"""Use point reward to learn where along the incoming flight to strike."""
from dataclasses import dataclass, field

import numpy as np

from .tt9_precontact_registration import RegistrationPlayer
from .tt12_online_timing import optical_role


OFFSETS = (0., .12, -.09)
SPREAD = np.array([.20, .10, .035, .035])


@dataclass
class StrikeMemory:
    observations: list = field(default_factory=list)

    def estimate(self, features, local):
        values = []
        for action in range(len(OFFSETS)):
            rows = [row for row in self.observations if row['action'] == action]
            if not rows:
                values.append(0.)
                continue
            points = np.array([row['features'] for row in rows])
            rewards = np.array([row['reward'] for row in rows])
            if local:
                distance = np.sum(((points - features) / SPREAD)**2, axis=1)
                weights = .25 + .75*np.exp(-.5*distance)
            else:
                weights = np.ones(len(rows))
            values.append(float(weights @ rewards / (2. + weights.sum())))
        return values

    def choose(self, features, mode):
        if mode == 'base':
            return 0
        values = self.estimate(features, local=mode != 'global')
        best = int(np.argmax(values))
        return best if values[best] > values[0] + .07 else 0


class OpticalCrossings:
    """Count renewed near-to-far travel using only the tracked RGB depth."""
    def __init__(self):
        self.armed = False
        self.clearances = []

    def observe(self, ball, tick):
        if ball is None or not ball.observed:
            return
        depth = ball.position[1]
        if depth < -.65:
            self.armed = True
        elif self.armed and depth > .25:
            self.clearances.append(tick)
            self.armed = False


def rally_value(reward, clearances):
    return float(.25*np.sign(reward) + .55*min(4,len(clearances)))


class StrikePlayer(RegistrationPlayer):
    def reset(self):
        super().reset()
        self.optical_role = None
        self.incoming = None
        self.crossings = OpticalCrossings()
        self.strike_choice = None

    def _stroke_goals(self, estimate):
        if self.strike_mode == 'optical':
            self.crossings.observe(self.scene.tracker.ball, self.scene.tick)
        if self.optical_role is None:
            self.optical_role = optical_role(self.scene.tracker.ball, self.scene.tick)
        self.incoming = estimate
        return super()._stroke_goals(estimate)

    def _reachable_plane(self, plane, ball):
        inherited = super()._reachable_plane(plane, ball)
        if self.optical_role != 'receive' or self.incoming is None:
            return inherited
        if self.strike_choice is None:
            velocity = self.incoming[1]
            features = [float(abs(ball[0]-self.adapter.ready[0])), float(ball[2]),
                        float(velocity[1]), float(velocity[2])]
            action = (self.practice_action if self.strike_mode == 'practice'
                      else self.strikes.choose(features, self.strike_mode))
            self.strike_choice = dict(action=action, features=features, tick=self.scene.tick)
        return inherited + OFFSETS[self.strike_choice['action']]

    def finish(self, reward):
        outcome = super().finish(reward)
        if self.strike_choice is not None and self.strike_mode in ('practice', 'online', 'optical'):
            value = (rally_value(reward, self.crossings.clearances) if self.strike_mode == 'optical'
                     else float(np.sign(reward)))
            row = dict(self.strike_choice, reward=value)
            self.strikes.observations.append(row)
        return dict(outcome, strike_choice=self.strike_choice, visual_crossings=self.crossings.clearances)
