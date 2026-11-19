"""Optional NumPy batch execution of the reference Pong timing filters."""

import math

import numpy as np

from pong_rapid_motor_recovery import (
    COMMIT_THRESHOLD, INVERSE_ORDERS, MEASUREMENT_VARIANCE, ORDERS,
    POSITION_LIMIT, PROCESS_NOISE, RESPONSE_SPEEDS, SPEED_LIMIT, motor_dynamics,
)
from pong_timing_controller import TimedMotorController


def observe_shadow_bank(shadows, requests, elapsed, position, *, dynamics=None):
    """Update each lag's motor belief, preserving its separate likelihood."""
    if dynamics is None:
        transition, response = motor_dynamics(elapsed)
    else:
        transition, response = dynamics
        response = response[:, None, :]
    probabilities = np.stack([belief.probabilities for belief in shadows])
    states = np.stack([belief.states for belief in shadows])
    covariances = np.stack([belief.covariances for belief in shadows])
    switches = np.stack([belief.switch for belief in shadows])
    joint = probabilities[:, :, None] * switches
    prior = joint.sum(axis=1)
    weights = (joint / prior[:, None, :]).swapaxes(1, 2)
    starting_states = weights @ states
    offsets = states[:, None, :, :] - starting_states[:, :, None, :]
    starting_covariances = (
        np.einsum('bij,bjkl->bikl', weights, covariances)
        + np.einsum('bij,bijk,bijl->bikl', weights, offsets, offsets))

    orders = INVERSE_ORDERS[np.arange(len(ORDERS))[None, :], np.asarray(requests)[:, None]]
    speeds = RESPONSE_SPEEDS[orders]
    states = starting_states @ transition.swapaxes(-1, -2) + speeds[:, :, None] * response
    np.clip(states[:, :, 1], -SPEED_LIMIT, SPEED_LIMIT, out=states[:, :, 1])
    stopped = abs(states[:, :, 0]) > POSITION_LIMIT
    states[stopped, 1] = 0.
    np.clip(states[:, :, 0], -POSITION_LIMIT, POSITION_LIMIT, out=states[:, :, 0])
    if dynamics is not None:
        transition = transition[:, None, :, :]
    covariances = transition @ starting_covariances @ transition.swapaxes(-1, -2) + PROCESS_NOISE
    covariances[stopped, 1, :] = 0.
    covariances[stopped, :, 1] = 0.
    covariances[stopped, 1, 1] = PROCESS_NOISE[1, 1]

    probabilities = prior
    evidence = np.zeros(len(shadows))
    if position is not None:
        innovation = position - states[:, :, 0]
        variance = covariances[:, :, 0, 0] + MEASUREMENT_VARIANCE
        standard_deviation = np.sqrt(variance)
        standardized = innovation / standard_deviation
        squared = np.square(standardized)
        loss = np.where(squared <= 9., squared, 6. * abs(standardized) - 9.)
        log_evidence = -.5 * (loss + np.log(variance))
        center = log_evidence.max(axis=1)
        posterior = prior * np.exp(log_evidence - center[:, None])
        mass = posterior.sum(axis=1)
        probabilities = posterior / mass[:, None]
        cross = covariances[:, :, :, 0].copy()
        bounded_innovation = np.clip(standardized, -3., 3.) * standard_deviation
        states += cross * (bounded_innovation / variance)[:, :, None]
        covariances -= cross[:, :, :, None] * cross[:, :, None, :] / variance[:, :, None, None]

        for index, best in enumerate(probabilities.argmax(axis=1)):
            if probabilities[index, best] >= COMMIT_THRESHOLD:
                shadows[index].confirmed = int(best)
        evidence = np.asarray([peak + math.log(total) for peak, total in zip(center, mass, strict=True)])

    for index, belief in enumerate(shadows):
        belief.states = states[index]
        belief.covariances = covariances[index]
        belief.probabilities = probabilities[index]
    return evidence


class BatchedTimedMotorController(TimedMotorController):
    """Run the original lag bank in one batch when timing inference is enabled."""

    def observe(self, *, own_hits, reward, terminated, truncated, elapsed_seconds, public_entities):
        if len(self.shadows) == 1:
            return super().observe(own_hits=own_hits, reward=reward, terminated=terminated,
                truncated=truncated, elapsed_seconds=elapsed_seconds, public_entities=public_entities)
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
        shadows = list(self.shadows.values())
        requests = []
        for lag, belief in self.shadows.items():
            if belief.states is None:
                belief.initialize(start_position)
            requests.append(self.requests[-lag - 1] if len(self.requests) > lag else belief.order[1])
        evidence = observe_shadow_bank(shadows, requests, elapsed_seconds, position)
        if self.timing_updates and position is not None:
            self._adapt_delay(evidence, elapsed_seconds)
