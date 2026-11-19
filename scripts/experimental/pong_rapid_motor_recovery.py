"""Adapt a visual servo with a bank of causal motor-response models."""

from functools import lru_cache
from itertools import permutations
import math

import numpy as np

from pong_relational_persistent_binding import PersistentRelationalController


ORDERS = tuple(permutations(range(3)))
INVERSE_ORDERS = np.argsort(ORDERS, axis=1)

# Fit on normal native request histories and observed controlled-entity positions.
# The constants describe the public image coordinate and observed camera timing.
RESPONSE_SECONDS = .0765
COAST_SECONDS = .2407
RESPONSE_SPEEDS = np.array([.7481, -.0013, -.7481])
SPEED_LIMIT = .6883
POSITION_LIMIT = .3997
MEASUREMENT_VARIANCE = .004 ** 2
PROCESS_NOISE = np.diag(np.square([.0005, .009, .013]))
INITIAL_UNCERTAINTY = np.diag(np.square([.004, .06, .12]))
CHANGE_PROBABILITY = .002
CHANGE_THRESHOLD = .10
COMMIT_THRESHOLD = .85


@lru_cache(maxsize=8)
def motor_dynamics(elapsed):
    """Discretize the two-stage command-to-motion response at camera time."""
    command_memory = math.exp(-elapsed / RESPONSE_SECONDS)
    velocity_memory = math.exp(-elapsed / COAST_SECONDS)
    scale = RESPONSE_SECONDS / (RESPONSE_SECONDS - COAST_SECONDS)
    command_to_velocity = scale * (command_memory - velocity_memory)
    velocity_to_position = COAST_SECONDS * (1 - velocity_memory)
    command_to_position = scale * (RESPONSE_SECONDS * (1 - command_memory) - velocity_to_position)
    transition = np.array([[1., velocity_to_position, command_to_position],
                           [0., velocity_memory, command_to_velocity],
                           [0., 0., command_memory]])
    response = np.array([elapsed - velocity_to_position - command_to_position,
                         1 - velocity_memory - command_to_velocity,
                         1 - command_memory])
    return transition, response


class VisualMotorBelief:
    """Estimate position, velocity, and lagged actuator response for each ordering."""

    def __init__(self, initial_order):
        count = len(ORDERS)
        self.switch = np.full((count, count), CHANGE_PROBABILITY / (count - 1))
        np.fill_diagonal(self.switch, 1 - CHANGE_PROBABILITY)
        self.probabilities = np.zeros(count)
        self.confirmed = ORDERS.index(tuple(initial_order))
        self.probabilities[self.confirmed] = 1.
        self.states = None
        self.covariances = None

    @property
    def order(self):
        return ORDERS[self.confirmed]

    def initialize(self, position):
        self.states = np.tile([position, 0., 0.], (len(ORDERS), 1))
        self.covariances = np.tile(INITIAL_UNCERTAINTY, (len(ORDERS), 1, 1))

    def observe(self, request, elapsed, position):
        # A changed map inherits the paddle's motion from the previous map.
        joint = self.probabilities[:, None] * self.switch
        prior = joint.sum(axis=0)
        weights = (joint / prior).T
        starting_states = weights @ self.states
        offsets = self.states[None, :, :] - starting_states[:, None, :]
        starting_covariances = (np.einsum('ij,jkl->ikl', weights, self.covariances)
                                + np.einsum('ij,ijk,ijl->ikl', weights, offsets, offsets))
        transition, response = motor_dynamics(elapsed)
        speeds = RESPONSE_SPEEDS[INVERSE_ORDERS[:, request]]
        self.states = starting_states @ transition.T + speeds[:, None] * response
        np.clip(self.states[:, 1], -SPEED_LIMIT, SPEED_LIMIT, out=self.states[:, 1])
        stopped = abs(self.states[:, 0]) > POSITION_LIMIT
        self.states[stopped, 1] = 0.
        np.clip(self.states[:, 0], -POSITION_LIMIT, POSITION_LIMIT, out=self.states[:, 0])
        self.covariances = transition @ starting_covariances @ transition.T + PROCESS_NOISE
        self.covariances[stopped, 1, :] = 0.
        self.covariances[stopped, :, 1] = 0.
        self.covariances[stopped, 1, 1] = PROCESS_NOISE[1, 1]

        self.probabilities = prior
        if position is None:
            return
        innovation = position - self.states[:, 0]
        variance = self.covariances[:, 0, 0] + MEASUREMENT_VARIANCE
        # Merged silhouettes sometimes shift the measured center. Temper the
        # score from a single camera so a displaced center cannot force a switch.
        standardized = innovation / np.sqrt(variance)
        squared = np.square(standardized)
        loss = np.where(squared <= 9., squared, 6. * abs(standardized) - 9.)
        log_evidence = -.5 * (loss + np.log(variance))
        posterior = prior * np.exp(log_evidence - log_evidence.max())
        self.probabilities = posterior / posterior.sum()
        cross = self.covariances[:, :, 0].copy()
        bounded_innovation = np.clip(standardized, -3., 3.) * np.sqrt(variance)
        self.states += cross * (bounded_innovation / variance)[:, None]
        self.covariances -= cross[:, :, None] * cross[:, None, :] / variance[:, None, None]

        best = int(self.probabilities.argmax())
        if self.probabilities[best] >= COMMIT_THRESHOLD:
            self.confirmed = best

    def request(self, intent, previous_request):
        if self.probabilities[self.confirmed] >= CHANGE_THRESHOLD:
            return self.order[intent]
        # A partly identified mapping can already have a clear up, down, or
        # coast request. Among unresolved choices, continue collecting a
        # measurable response from the previous request when it remains useful.
        effect = self.probabilities @ (INVERSE_ORDERS == intent)
        possible = np.flatnonzero(effect >= effect.max() - .04)
        if previous_request in possible:
            return previous_request
        return int(possible[0])


class RapidMotorController:
    """Keep the inherited interception intent through an unseen remapping."""

    def __init__(self, policy, learner):
        self.inherited = PersistentRelationalController(policy)
        self.belief = VisualMotorBelief([learner.request(intent) for intent in range(3)])
        self.adaptation_enabled = learner.updates_enabled
        self.axis = policy.original.axis
        self.motor_id = None
        self.motor_station = None
        self.pending = None
        self.last_request = None

    @property
    def effective_requests(self):
        return list(self.belief.order)

    def _position(self, entities):
        candidates = [entity for entity in entities if entity['observed']
            and abs(entity['normalized_position'][1 - self.axis] - self.motor_station) < 1 / 32]
        if not candidates:
            return None
        motor = min(candidates, key=lambda entity: (
            entity['id'] != self.motor_id,
            abs(entity['normalized_position'][1 - self.axis] - self.motor_station)))
        self.motor_id = motor['id']
        return float(motor['normalized_position'][self.axis])

    def choose(self, entities, baseline_action):
        intent, decision = self.inherited.choose(entities, baseline_action)
        if self.adaptation_enabled and self.motor_station is None and decision['bound_ids'] is not None:
            self.motor_id = decision['bound_ids'][0]
            motor = next((item for item in entities
                          if item['id'] == self.motor_id and item['observed']), None)
            if motor is not None:
                self.motor_station = float(motor['normalized_position'][1 - self.axis])

        if self.motor_station is not None and self.belief.states is None:
            position = self._position(entities)
            if position is not None:
                self.belief.initialize(position)
        request = self.belief.request(intent, self.last_request)
        self.pending = request
        self.last_request = request
        return request, dict(intended=intent, action=request, probing=False,
                             effective_requests=self.effective_requests)

    def observe(self, *, own_hits, reward, terminated, truncated, elapsed_seconds, public_entities):
        if self.pending is None or not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            raise ValueError('a requested action and positive public elapsed time are required')
        self.inherited.observe(own_hits, terminated)
        if self.belief.states is not None:
            self.belief.observe(self.pending, elapsed_seconds, self._position(public_entities))
        self.pending = None
