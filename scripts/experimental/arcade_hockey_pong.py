"""Camera-only Pong pursuit and an experience prior for unseen wall responses."""

from dataclasses import dataclass
import math

import numpy as np

from arcade_wall_learn import WallFlight, project_crossing
from pong_rapid_motor_recovery import COAST_SECONDS, RESPONSE_SECONDS
from pong_timing_controller import TimingMotorBelief


PONG_WALLS = (1 / 64, 63 / 64)
PONG_CONTACT = 4 / 64
COEFFICIENTS = np.linspace(0., 1., 101)


def alternatives(anchor, blocks):
    """Infer how often wall responses differed and which changes were observed."""
    broad = np.full(len(COEFFICIENTS), 1 / len(COEFFICIENTS))
    kernels = []
    familiar = 0
    for memory in blocks:
        error = memory.uncertainty
        if memory.count < 2:
            continue
        if abs(anchor - memory.coefficient) < 2.5 * math.hypot(error, .025):
            familiar += 1
            continue
        width = math.hypot(.085, error)
        density = np.exp(-.5 * ((COEFFICIENTS - memory.coefficient) / width)**2)
        kernels.append(density / density.sum())
    rate = (1 + len(kernels)) / (familiar + len(kernels) + 2)
    return (.25 * broad + .75 * np.mean(kernels, axis=0) if kernels else broad), rate


class AdaptiveWall:
    """Keep familiar play until a surprising camera bounce supports a new context."""

    def __init__(self, anchor, practice=(), mode='live'):
        if mode not in ('live', 'fixed_on_surprise', 'frozen'):
            raise ValueError('unknown wall update mode')
        self.anchor = anchor.coefficient
        self.anchor_error = max(.025, anchor.uncertainty or .05)
        self.prior, self.change_rate = alternatives(self.anchor, practice)
        self.log_odds = math.log(self.change_rate / (1. - self.change_rate))
        self.mode = mode
        self.coefficient = self.anchor
        self.log_prior = np.log(self.prior)
        self.log_mass = self.log_prior.copy()
        self.changed = False
        self.observations = 0
        self.change_probability = self.change_rate

    def _log_likelihood_ratio(self, ratio, uncertainty):
        familiar_error = math.hypot(uncertainty, self.anchor_error)
        familiar = -.5 * ((self.anchor - ratio) / familiar_error)**2 - math.log(familiar_error)
        return -.5 * ((COEFFICIENTS - ratio) / uncertainty)**2 - math.log(uncertainty) - familiar

    def _ongoing_or_new(self):
        # A blend with a proposed new surface allows a real change after
        # several observations that already established the familiar response.
        return self.log_mass if self.changed else np.logaddexp(self.log_mass, self.log_prior) - math.log(2.)

    def _read(self, log_mass, *, commit):
        center = float(log_mass.max())
        probability = np.exp(log_mass - center)
        mass = float(probability.sum())
        log_odds = self.log_odds + center + math.log(mass)
        self.change_probability = 1 / (1 + math.exp(-max(-680., min(680., log_odds))))
        changed = self.changed or self.change_probability >= .82
        if commit:
            self.changed = changed
        if changed and self.mode == 'fixed_on_surprise':
            self.coefficient = float(self.prior @ COEFFICIENTS)
        elif changed:
            self.coefficient = float(probability @ COEFFICIENTS / mass)
        else:
            self.coefficient = self.anchor

    def observe(self, ratio, uncertainty):
        self.observations += 1
        if self.mode == 'frozen':
            return
        self.log_mass = self._ongoing_or_new() + self._log_likelihood_ratio(ratio, uncertainty)
        self._read(self.log_mass, commit=True)

    def observe_hint(self, ratio, uncertainty):
        if self.mode != 'frozen':
            # Overlapping short windows supersede each other. Only the full
            # post-bounce estimate becomes a persistent likelihood term.
            self._read(self._ongoing_or_new() + self._log_likelihood_ratio(ratio, uncertainty), commit=False)


@dataclass(frozen=True)
class PongImage:
    paddle: float
    ball: tuple[float, float] | None

    @classmethod
    def read(cls, frame):
        image = np.asarray(frame)
        if image.dtype != np.bool_ or image.shape != (64, 64):
            raise ValueError('a native binary Pong camera frame is required')
        rows = np.flatnonzero(image[:, 3])
        if not len(rows):
            raise ValueError('the visible left paddle is required')
        paddle = (rows[0] + rows[-1] + 1.) / 128
        y, x = np.nonzero(image[:, 5:59])
        ball = ((x.mean() + 5.5) / 64, (y.mean() + .5) / 64) if len(x) else None
        return cls(float(paddle), ball)


class PongFlightActor:
    """Use the learned Pong actuator map with the current public flight forecast."""

    def __init__(self, timing_checkpoint, memory):
        self.timing = TimingMotorBelief(timing_checkpoint['motor_order'])
        self.timing.probabilities = np.asarray(timing_checkpoint['motor_probabilities']).copy()
        self.flight = WallFlight(memory, PONG_WALLS, position_noise=.0048, hints=isinstance(memory, AdaptiveWall))
        self.paddle = .5
        self.pending = None
        self.target = None
        self.last_target = None

    def start(self, frame):
        current = PongImage.read(frame)
        self.timing.initialize(current.paddle - .5)
        self.flight.observe(current.ball, .02)
        self.paddle = current.paddle

    def choose(self):
        position = self.flight.position
        velocity = self.flight.velocity
        self.target = None
        if position is not None and velocity is not None and velocity[0] < -.10:
            self.target = project_crossing(position, velocity, PONG_CONTACT, PONG_WALLS,
                                           self.flight.memory.coefficient)
        if self.target is not None:
            self.last_target = self.target
        elif position is not None:
            self.last_target = None

        state = self.timing.states[self.timing.confirmed]
        aim = .5 if self.last_target is None else min(.90, max(.10, self.last_target))
        stop = self.paddle + COAST_SECONDS * state[1] + RESPONSE_SECONDS * state[2]
        error = aim - stop
        intent = 0 if error > .014 else 2 if error < -.014 else 1
        self.pending = self.timing.request(intent, self.pending)
        return self.pending

    def observe(self, frame, elapsed):
        current = PongImage.read(frame)
        self.timing.observe(self.pending, elapsed, current.paddle - .5)
        self.paddle = current.paddle
        bounce = self.flight.observe(current.ball, elapsed)
        hint = self.flight.last_hint
        if hint is not None:
            self.flight.memory.observe_hint(hint.coefficient, hint.uncertainty)
        return bounce
