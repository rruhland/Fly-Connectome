"""Infer actuator rise time separately from request delay in visual Pong."""

from functools import lru_cache
import math

import numpy as np

from pong_rapid_motor_recovery import (
    CHANGE_THRESHOLD, COAST_SECONDS, INVERSE_ORDERS, POSITION_LIMIT, RESPONSE_SECONDS,
    RESPONSE_SPEEDS, SPEED_LIMIT, motor_dynamics,
)
from pong_timing_controller import (
    EVIDENCE_CONFIRM_CAMERAS, EVIDENCE_DECAY_SECONDS, EVIDENCE_MARGIN,
    EVIDENCE_MEMORY_LIMIT, TimedMotorController, TimingMotorBelief,
)
from pong_timing_fast import observe_shadow_bank


RESPONSE_TIMES = (RESPONSE_SECONDS, .125, .18, .28, .42, .63)


@lru_cache(maxsize=512)
def response_dynamics(elapsed, response_seconds):
    """Return the learned coast model with a variable command response time."""
    if response_seconds == RESPONSE_SECONDS:
        return motor_dynamics(elapsed)
    response_memory = math.exp(-elapsed / response_seconds)
    velocity_memory = math.exp(-elapsed / COAST_SECONDS)
    velocity_position = COAST_SECONDS * (1 - velocity_memory)
    if abs(response_seconds - COAST_SECONDS) < 1e-7:
        activation_velocity = elapsed * velocity_memory / COAST_SECONDS
        activation_position = velocity_position - elapsed * velocity_memory
    else:
        scale = response_seconds / (response_seconds - COAST_SECONDS)
        activation_velocity = scale * (response_memory - velocity_memory)
        activation_position = scale * (response_seconds * (1 - response_memory) - velocity_position)
    transition = np.array([[1., velocity_position, activation_position],
                           [0., velocity_memory, activation_velocity],
                           [0., 0., response_memory]])
    drive = np.array([elapsed - velocity_position - activation_position,
                      1 - velocity_memory - activation_velocity,
                      1 - response_memory])
    return transition, drive


@lru_cache(maxsize=48)
def response_bank_dynamics(elapsed, hypotheses):
    transitions, drives = zip(*(response_dynamics(elapsed, response) for response, _ in hypotheses))
    return np.stack(transitions), np.stack(drives)


def motor_projection(state, request_speed, duration, response_seconds):
    transition, drive = response_dynamics(duration, response_seconds)
    projected = transition @ state + request_speed * drive
    projected[1] = np.clip(projected[1], -SPEED_LIMIT, SPEED_LIMIT)
    if abs(projected[0]) > POSITION_LIMIT:
        projected[0] = np.clip(projected[0], -POSITION_LIMIT, POSITION_LIMIT)
        projected[1] = 0.
    return projected


class ActuatorResponseController(TimedMotorController):
    """Compare both delivery delays and actuator rise times using camera innovations."""

    def __init__(self, policy, learner, *, timing_state=None, timing_updates=True,
                 response_updates=True):
        super().__init__(policy, learner, timing_state=timing_state, timing_updates=timing_updates)
        stored_response = RESPONSE_SECONDS if timing_state is None else timing_state.get(
            'response_hypothesis', RESPONSE_SECONDS)
        self.response_hypothesis = float(stored_response)
        if self.response_hypothesis not in RESPONSE_TIMES:
            raise ValueError('unrecognized actuator response hypothesis')
        self.response_updates = bool(response_updates)
        self.response_times = RESPONSE_TIMES if self.response_updates else (self.response_hypothesis,)
        self.hypotheses = tuple((response, lag) for response in self.response_times for lag in self.lags)
        self.banks = {self.response_hypothesis: self.shadows}
        for response in self.response_times:
            if response == self.response_hypothesis:
                continue
            self.banks[response] = {lag: TimingMotorBelief(self.belief.order) for lag in self.lags}
            for belief in self.banks[response].values():
                belief.probabilities = self.belief.probabilities.copy()
        self.evidence = np.zeros(len(self.hypotheses))
        self.response_seconds = self.response_hypothesis if timing_state is None else float(
            timing_state.get('response_seconds', self.response_hypothesis))
        if not RESPONSE_TIMES[0] <= self.response_seconds <= RESPONSE_TIMES[-1]:
            raise ValueError('response time estimate lies outside the supported interval')
        self.candidate_response = None
        self.response_changes = 0

    def state(self):
        state = super().state()
        state.update(response_hypothesis=self.response_hypothesis,
                     response_seconds=self.response_seconds)
        return state

    def observe(self, *, own_hits, reward, terminated, truncated, elapsed_seconds, public_entities):
        if self.pending is None or not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            raise ValueError('a requested action and positive public elapsed time are required')
        self.inherited.observe(own_hits, terminated)
        self.requests.append(self.pending)
        self.pending = None
        self.camera_intervals.append(elapsed_seconds)
        self.seconds += elapsed_seconds
        if self.belief.states is None:
            return

        position = self._position(public_entities)
        start_position = float(self.belief.states[self.belief.confirmed, 0])
        shadows = []
        requests = []
        for response, lag in self.hypotheses:
            shadow = self.banks[response][lag]
            if shadow.states is None:
                shadow.initialize(start_position)
            shadows.append(shadow)
            requests.append(self.requests[-lag - 1] if len(self.requests) > lag else shadow.order[1])
        observations = observe_shadow_bank(shadows, requests, elapsed_seconds, position,
            dynamics=response_bank_dynamics(elapsed_seconds, self.hypotheses))
        if position is not None and len(self.hypotheses) > 1:
            self._adapt_response(observations, elapsed_seconds)

    def _adapt_response(self, observations, elapsed):
        self.evidence *= math.exp(-elapsed / EVIDENCE_DECAY_SECONDS)
        self.evidence += np.clip(observations - observations.max(), -2., 0.)
        self.evidence -= self.evidence.max()
        np.maximum(self.evidence, -EVIDENCE_MEMORY_LIMIT, out=self.evidence)
        current = self.response_hypothesis, self.delay_commands
        preference = [.05 * abs(lag - self.delay_commands)
                      + .25 * abs(math.log(response / self.response_hypothesis))
                      for response, lag in self.hypotheses]
        adjusted = self.evidence - preference
        best = int(adjusted.argmax())
        candidate = self.hypotheses[best]
        active = self.hypotheses.index(current)
        if adjusted[best] - adjusted[active] < EVIDENCE_MARGIN:
            self.candidate_response, self.candidate_dwell = None, 0
        else:
            self.candidate_dwell = self.candidate_dwell + 1 if candidate == self.candidate_response else 1
            self.candidate_response = candidate
            if self.candidate_dwell >= EVIDENCE_CONFIRM_CAMERAS:
                self.response_hypothesis, self.delay_commands = candidate
                self.shadows = self.banks[self.response_hypothesis]
                self.belief = self.shadows[self.delay_commands]
                self.response_changes += int(current[0] != self.response_hypothesis)
                self.delay_changes += int(current[1] != self.delay_commands)
                self.candidate_response, self.candidate_dwell = None, 0
        # Read a continuous response estimate from nearby hypotheses. Keep the
        # acquired fast model exact when it is the confirmed explanation.
        if not self.response_updates:
            return
        if self.response_hypothesis == RESPONSE_SECONDS:
            self.response_seconds = RESPONSE_SECONDS
        else:
            column = self.lags.index(self.delay_commands)
            local_evidence = self.evidence[column::len(self.lags)]
            weights = np.exp(local_evidence - local_evidence.max())
            estimate = math.exp(float(weights @ np.log(self.response_times) / weights.sum()))
            self.response_seconds = round(estimate, 3)

    def _queued_state(self):
        belief = self.belief
        best = belief.confirmed
        if belief.probabilities[best] < CHANGE_THRESHOLD:
            best = int(belief.probabilities.argmax())
        state = belief.states[best].copy()
        queued = list(self.requests)[-self.delay_commands:] if self.delay_commands else []
        queued = [belief.order[1]] * (self.delay_commands - len(queued)) + queued
        for request in queued:
            speed = RESPONSE_SPEEDS[INVERSE_ORDERS[best, request]]
            state = motor_projection(state, speed, self.camera_seconds, self.response_seconds)
        return state

    def choose(self, entities, baseline_action):
        previous_request = self.last_request
        request, decision = super().choose(entities, baseline_action)
        decision['response_seconds'] = self.response_seconds
        # Preserve the ordinary and fast-response FIFO paths. For a slower
        # response, choose against the residual travel already in the actuator.
        if self.response_hypothesis == RESPONSE_SECONDS or self.belief.states is None:
            return request, decision
        context = self.arrival_context
        if context is None or self.contact_offset != 0.:
            return request, decision
        horizon = context.seconds - self.delay_commands * self.camera_seconds
        if horizon <= 0:
            return request, decision
        queued = self._queued_state()
        projected = motor_projection(queued, RESPONSE_SPEEDS[1], round(min(horizon, .85), 3),
                                     self.response_seconds)
        target = float(np.clip(context.intercept, context.lower, context.upper))
        error = target - projected[0]
        intent = 1 if abs(error) < .006 else (0 if error > 0 else 2)
        request = self.belief.request(intent, previous_request)
        self.pending = self.last_request = request
        self.inherited.pending_action = intent
        decision.update(action=request, intended=intent)
        return request, decision
