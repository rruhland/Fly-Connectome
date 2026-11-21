"""Choose the actuator response for each stereo-recognized service role."""

from dataclasses import asdict, dataclass, field
from math import log, sqrt

from .tt9_precontact_registration import RegistrationPlayer
from .tt11_timing_select import MODES


ROLES = ('receive', 'serve')
OPENING_FRAMES = 8
SAMPLES_PER_MODEL = 2


def _empty_counts():
    return {role: dict.fromkeys(MODES, 0) for role in ROLES}


@dataclass
class TimingChoice:
    anchor: str
    attempts: dict = field(default_factory=_empty_counts)
    scored: dict = field(default_factory=_empty_counts)
    wins: dict = field(default_factory=_empty_counts)

    def values(self, role):
        return {mode: (self.wins[role][mode] + 1) / (self.scored[role][mode] + 2) for mode in MODES}

    def preferred(self, role):
        values = self.values(role)
        return max(MODES, key=lambda mode: (values[mode], mode == self.anchor))

    def choose(self, role):
        tries = self.attempts[role]
        if min(tries.values()) < SAMPLES_PER_MODEL:
            return min(MODES, key=lambda mode: (tries[mode], mode != self.anchor)), 'sample'
        values = self.values(role)
        total = sum(tries.values())
        return max(MODES, key=lambda mode: (values[mode] + .20 * sqrt(log(1 + total) / (1 + tries[mode])),
                                            mode == self.anchor)), 'reward'

    def observe(self, role, mode, reward):
        self.attempts[role][mode] += 1
        if reward != 0:
            self.scored[role][mode] += 1
            self.wins[role][mode] += int(reward > 0)

    def state(self):
        return asdict(self)


def optical_role(ball, tick):
    if tick >= OPENING_FRAMES or ball is None or not ball.observed:
        return None
    depth = ball.position[1]
    if depth < -1.10:
        return 'serve'
    if depth > .35:
        return 'receive'
    return None


class OnlineTimingPlayer(RegistrationPlayer):
    """Share one registered player across timing hypotheses and switch at the opening sighting."""

    def __init__(self, *args, models, choice, policy, **kwargs):
        self.models = models
        self.choice = choice
        self.policy = policy
        super().__init__(*args, **kwargs)

    def reset(self):
        self.adapter = self.models[self.choice.anchor]
        super().reset()
        self.opening = []
        self.timing_decision = None

    def update(self, frames, preceding=None):
        self.preceding = preceding
        return super().update(frames, preceding)

    def _ball(self, tracks, actor):
        ball = super()._ball(tracks, actor)
        tick = self.scene.tick
        if self.timing_decision is not None or tick >= OPENING_FRAMES:
            self.opening.clear()
            return ball
        pitch = getattr(self.scene.tracker, 'actor_pitch', None) if actor is not None and actor.observed else None
        self.opening.append((self.preceding, actor, pitch))
        role = optical_role(ball, tick)
        if role is None:
            return ball

        if self.policy == 'adaptive':
            mode, rule = self.choice.choose(role)
        elif self.policy == 'retained':
            mode, rule = self.choice.preferred(role), 'retained'
        else:
            mode, rule = self.policy, 'fixed'
        self.timing_decision = dict(role=role, mode=mode, rule=rule, tick=tick, depth=float(ball.position[1]),
                                    attempts=self.choice.attempts[role].copy(), values=self.choice.values(role))
        if mode != self.choice.anchor:
            self.adapter = self.models[mode]
            self.servo = self._make_servo()
            # Reconstruct the estimator under the selected curve from the same
            # preceding commands and RGB blade sightings, without rerunning physics.
            for preceding, blade, pitch in self.opening:
                self.servo.advance(preceding)
                self.servo.observe(blade, pitch)
        self.opening.clear()
        return ball

    def finish(self, reward):
        details = super().finish(reward)
        decision = self.timing_decision
        if decision is not None and self.policy == 'adaptive':
            self.choice.observe(decision['role'], decision['mode'], reward)
        return dict(details, timing_choice=decision)
