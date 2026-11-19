"""Learn request latency from the public visual stream and anticipate queued motion."""

from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np

from pong_rapid_motor_recovery import (
    COMMIT_THRESHOLD, INVERSE_ORDERS, MEASUREMENT_VARIANCE, ORDERS,
    POSITION_LIMIT, RESPONSE_SPEEDS, SPEED_LIMIT, RapidMotorController,
    VisualMotorBelief, motor_dynamics,
)
from pong_relational_persistent_binding import PersistentRelationalController


DELAYS = tuple(range(0, 25, 3))
NOMINAL_CAMERA_SECONDS = .02
EVIDENCE_DECAY_SECONDS = .65
EVIDENCE_MARGIN = 4.5
EVIDENCE_MEMORY_LIMIT = 7.
EVIDENCE_CONFIRM_CAMERAS = 4
EARLY_MOVEMENT = 1 / 64
SCHEMA = 'public-visual-pong-timing-v1'


class TimingMotorBelief(VisualMotorBelief):
    """Use Stage 1 motion predictions and return their marginal visual evidence."""

    def observe(self, request, elapsed, position):
        super().observe(request, elapsed, None)
        if position is None:
            return 0.
        innovation = position - self.states[:, 0]
        variance = self.covariances[:, 0, 0] + MEASUREMENT_VARIANCE
        standardized = innovation / np.sqrt(variance)
        squared = np.square(standardized)
        loss = np.where(squared <= 9., squared, 6. * abs(standardized) - 9.)
        log_evidence = -.5 * (loss + np.log(variance))
        center = log_evidence.max()
        posterior = self.probabilities * np.exp(log_evidence - center)
        mass = posterior.sum()
        self.probabilities = posterior / mass
        cross = self.covariances[:, :, 0].copy()
        bounded_innovation = np.clip(standardized, -3., 3.) * np.sqrt(variance)
        self.states += cross * (bounded_innovation / variance)[:, None]
        self.covariances -= cross[:, :, None] * cross[:, None, :] / variance[:, None, None]
        best = int(self.probabilities.argmax())
        if self.probabilities[best] >= COMMIT_THRESHOLD:
            self.confirmed = best
        return float(center + math.log(mass))


@dataclass(frozen=True)
class ArrivalContext:
    intercept: float
    seconds: float
    paddle: float
    lower: float
    upper: float
    reach: float


def reflect(coordinate, boundary=.5):
    return boundary - abs((coordinate + boundary) % (4 * boundary) - 2 * boundary)


class TimingIntent(PersistentRelationalController):
    """Evaluate the same learned relation at the time a new request can arrive."""

    def __init__(self, policy, controller):
        super().__init__(policy)
        self.controller = controller
        self.history = deque(maxlen=8)
        self.binding = None
        self.arrival_context = None

    def choose(self, entities, baseline_action):
        action, decision = super().choose(entities, baseline_action)
        self.arrival_context = None
        binding = self.bound_ids
        if binding != self.binding:
            self.history.clear()
            self.binding = binding
        axis = self.policy.original.axis
        if binding is None or not decision['supported']:
            if callable(self.controller.contact_offset):
                self.controller.contact_offset(None)
            return action, decision
        lookup = {entity['id']: entity for entity in entities}
        own, target = (lookup[item] for item in binding)
        own_position = own['normalized_position'][axis]
        along = target['normalized_position'][axis]
        across = target['normalized_position'][1 - axis]
        time = self.controller.seconds
        along_speed = across_speed = 0.
        if target['observed']:
            if self.history and time - self.history[-1][0] > .10:
                self.history.clear()
            self.history.append((time, along, across))
        if len(self.history) >= 4:
            old_time, old_along, old_across = self.history[0]
            recent_time, recent_along, recent_across = self.history[-1]
            elapsed = recent_time - old_time
            if elapsed > 0 and time - recent_time <= .10:
                along_speed = (recent_along - old_along) / elapsed
                across_speed = (recent_across - old_across) / elapsed
        if .12 < abs(across_speed) < .9 and abs(along_speed) < .75:
            seconds = (own['normalized_position'][1 - axis] - across) / across_speed
            if 0 <= seconds <= 3.:
                intercept = reflect(along + seconds * along_speed)
                self.arrival_context = ArrivalContext(intercept, seconds, own_position,
                    -POSITION_LIMIT, POSITION_LIMIT, abs(intercept - own_position))

        lead = self.controller.delay_commands * self.controller.camera_seconds
        offset = self.controller.contact_offset
        if callable(offset):
            offset = float(offset(self.arrival_context))
        if not math.isfinite(offset):
            raise ValueError('finite normalized contact offset required')
        if not lead and not offset:
            return action, decision
        target_lead = along_speed * lead
        if self.arrival_context is not None:
            aim = min(POSITION_LIMIT, max(-POSITION_LIMIT, self.arrival_context.intercept + offset))
            target_lead = aim - along
            offset = 0.
        paddle_lead = self.controller.paddle_lead(brake=True)
        original = self.policy.original
        score = decision['score'] + original.polarity * (
            target_lead - paddle_lead + offset) / original.scales[axis]
        deadzone = decision['deadzone']
        action = 2 if score > deadzone else 0 if score < -deadzone else 1
        self.pending_action = action
        decision.update(action=action, score=score)
        return action, decision


class TimedMotorController(RapidMotorController):
    """Carry the selected timing and Stage 1 button belief across rallies."""

    def __init__(self, policy, learner, *, timing_state=None, timing_updates=True):
        super().__init__(policy, learner)
        self.delay_commands = 0
        self.timing_updates = bool(timing_updates)
        if timing_state is not None:
            if (timing_state.get('schema') != SCHEMA
                    or timing_state.get('delay_commands') not in DELAYS):
                raise ValueError('invalid carried timing state')
            self.delay_commands = timing_state['delay_commands']
            order = tuple(timing_state['motor_order'])
            probabilities = np.asarray(timing_state['motor_probabilities'], dtype=float)
            if (order not in ORDERS or probabilities.shape != (len(ORDERS),)
                    or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                    or not math.isclose(float(probabilities.sum()), 1.)):
                raise ValueError('invalid carried motor state')
            self.belief.confirmed = ORDERS.index(order)
            self.belief.probabilities = probabilities.copy()
        self.lags = DELAYS if self.timing_updates else (self.delay_commands,)
        self.shadows = {lag: TimingMotorBelief(self.belief.order) for lag in self.lags}
        for belief in self.shadows.values():
            belief.probabilities = self.belief.probabilities.copy()
        self.belief = self.shadows[self.delay_commands]
        self.evidence = np.zeros(len(self.lags))
        self.candidate = None
        self.candidate_dwell = 0
        self.delay_changes = 0
        self.carried_delay = self.delay_commands
        self.initial_idle = self.belief.order[1]
        self.first_motion_request = None
        self.initial_position = None
        self.requests = deque(maxlen=max(DELAYS) + 1)
        self.camera_intervals = deque(maxlen=5)
        self.seconds = 0.
        self.contact_offset = 0.
        self.inherited = TimingIntent(policy, self)

    @property
    def arrival_context(self):
        return self.inherited.arrival_context

    @property
    def camera_seconds(self):
        if not self.camera_intervals:
            return NOMINAL_CAMERA_SECONDS
        return round(sum(self.camera_intervals) / len(self.camera_intervals), 8)

    def state(self):
        return dict(schema=SCHEMA, delay_commands=self.delay_commands,
                    motor_order=list(self.belief.order),
                    motor_probabilities=self.belief.probabilities.tolist())

    @classmethod
    def carried_motor(cls, controller):
        """Promote a settled Stage 1 controller without relearning its button order."""
        return dict(schema=SCHEMA, delay_commands=0,
                    motor_order=list(controller.belief.order),
                    motor_probabilities=controller.belief.probabilities.tolist())

    def choose(self, entities, baseline_action):
        action, decision = super().choose(entities, baseline_action)
        context = self.arrival_context
        decision.update(delay_commands=self.delay_commands,
                        arrival_context=asdict(context) if context else None)
        return action, decision

    def paddle_lead(self, *, brake=False):
        """Project over already requested commands using public camera durations."""
        if self.belief.states is None:
            return 0.
        best = self.belief.confirmed
        if self.belief.probabilities[best] < .10:
            best = int(self.belief.probabilities.argmax())
        state = self.belief.states[best].copy()
        position = float(state[0])
        queued = list(self.requests)[-self.delay_commands:] if self.delay_commands else []
        idle = self.belief.order[1]
        queued = [idle] * (self.delay_commands - len(queued)) + queued
        elapsed = self.camera_seconds
        transition, response = motor_dynamics(elapsed)
        for request in queued:
            speed = RESPONSE_SPEEDS[INVERSE_ORDERS[best, request]]
            state = transition @ state + speed * response
            state[1] = max(-SPEED_LIMIT, min(SPEED_LIMIT, state[1]))
            if abs(state[0]) > POSITION_LIMIT:
                state[0] = math.copysign(POSITION_LIMIT, state[0])
                state[1] = 0.
        if brake:
            state[0] += .14 * state[1] + .018 * state[2]
            state[0] = max(-POSITION_LIMIT, min(POSITION_LIMIT, state[0]))
        return float(state[0]) - position

    def observe(self, *, own_hits, reward, terminated, truncated, elapsed_seconds, public_entities):
        if self.pending is None or not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            raise ValueError('a requested action and positive public elapsed time are required')
        self.inherited.observe(own_hits, terminated)
        self.requests.append(self.pending)
        ordinal = self.inherited.ordinal - 1
        if self.first_motion_request is None and self.pending != self.initial_idle:
            self.first_motion_request = ordinal
        self.pending = None
        self.camera_intervals.append(elapsed_seconds)
        self.seconds += elapsed_seconds
        if self.belief.states is None:
            return
        start_position = float(self.belief.states[self.belief.confirmed, 0])
        if self.initial_position is None:
            self.initial_position = (self.motor_id, start_position)
        position = self._position(public_entities)
        observations = []
        for lag, belief in self.shadows.items():
            if belief.states is None:
                belief.initialize(start_position)
            request = (self.requests[-lag - 1] if len(self.requests) > lag
                       else belief.order[1])
            observations.append(belief.observe(request, elapsed_seconds, position))
        if self.timing_updates and position is not None:
            self._adapt_delay(np.asarray(observations), elapsed_seconds)
            self._check_early_motion(ordinal, position)

    def _check_early_motion(self, ordinal, position):
        """Reconsider carried lag when fresh paddle motion precedes any queued drive."""
        if (not self.carried_delay or self.delay_commands != self.carried_delay
                or self.first_motion_request is None):
            return
        age = ordinal - self.first_motion_request
        identity, origin = self.initial_position
        if (age < self.carried_delay and self.motor_id == identity
                and abs(position - origin) >= EARLY_MOVEMENT):
            # Its complete observed request history contains no matured movement
            # at the carried lag. Use the immediate model provisionally while
            # the bank continues comparing every candidate from camera motion.
            self.evidence[self.lags.index(self.carried_delay)] = -EVIDENCE_MEMORY_LIMIT
            self._select_delay(0)
            self.carried_delay = 0

    def _select_delay(self, candidate):
        self.delay_commands = candidate
        self.belief = self.shadows[candidate]
        self.delay_changes += 1
        self.candidate, self.candidate_dwell = None, 0

    def _adapt_delay(self, observations, elapsed):
        self.evidence *= math.exp(-elapsed / EVIDENCE_DECAY_SECONDS)
        # Cap the influence of a single merged or misidentified silhouette.
        self.evidence += np.clip(observations - observations.max(), -2., 0.)
        self.evidence -= self.evidence.max()
        np.maximum(self.evidence, -EVIDENCE_MEMORY_LIMIT, out=self.evidence)
        preference = np.abs(np.asarray(self.lags) - self.delay_commands) * .05
        adjusted = self.evidence - preference
        best = int(adjusted.argmax())
        active = self.lags.index(self.delay_commands)
        candidate = self.lags[best]
        if adjusted[best] - adjusted[active] < EVIDENCE_MARGIN:
            self.candidate, self.candidate_dwell = None, 0
            return
        self.candidate_dwell = self.candidate_dwell + 1 if candidate == self.candidate else 1
        self.candidate = candidate
        if self.candidate_dwell >= EVIDENCE_CONFIRM_CAMERAS:
            self._select_delay(candidate)
