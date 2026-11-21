"""Fit a changing-delay model from the camera's competing motor predictions."""

from collections import deque

import numpy as np


HYPOTHESES = np.arange(6)
BASE_TRANSITIONS = 24. * np.eye(len(HYPOTHESES)) + .3


def optical_likelihood(sighting, previous, predictions, offset, noise):
    """Compare alternative delayed motor predictions with public camera displacement."""
    if sighting is None:
        return None
    measured = np.asarray(sighting)
    predictions = np.asarray(predictions)
    residual = (measured - predictions - offset) / noise
    if np.max(np.ptp(predictions / noise, axis=0)) <= .65 or np.min(np.linalg.norm(residual, axis=1)) >= 7.:
        return None
    loss = .65 * np.sum(np.minimum(residual**2, 9.), axis=1)
    if previous is not None:
        measured_before, predicted_before = previous
        movement = (measured - measured_before - predictions + predicted_before) / (1.4 * noise)
        loss += np.sum(np.minimum(movement**2, 9.), axis=1)
    return np.exp(-.5 * (loss - loss.min()))


def normalized_rows(values):
    return values / values.sum(axis=1, keepdims=True)


def expected_counts(emissions, initial, transition):
    """Forward-backward expected transition counts from camera likelihoods."""
    history = np.asarray(emissions)
    forwards = np.empty_like(history)
    state = initial
    for i, likelihood in enumerate(history):
        state = state * likelihood
        state /= state.sum()
        forwards[i] = state
        state = state @ transition
    counts = np.zeros_like(transition)
    backwards = np.ones_like(initial)
    for forward, next_emission in zip(forwards[-2::-1], history[:0:-1], strict=True):
        weighted = transition * (next_emission * backwards)
        pair = forward[:, None] * weighted
        counts += pair / pair.sum()
        backwards = weighted.sum(axis=1)
        backwards /= backwards.sum()
    first = forwards[0] * backwards
    return counts, first / first.sum()


def train_dynamics(receipt):
    """Learn a Markov transition prior with regularized expectation maximization."""
    records = [point['optical_likelihoods'] for visit in receipt['visits'] for point in visit['points']]
    emissions = [[np.ones(len(HYPOTHESES)) if row is None else row for row in point] for point in records]
    initial = np.full(len(HYPOTHESES), 1. / len(HYPOTHESES))
    transition = normalized_rows(BASE_TRANSITIONS)
    for _ in range(16):
        counts = BASE_TRANSITIONS.copy()
        start = np.ones_like(initial)
        for point in emissions:
            used, first = expected_counts(point, initial, transition)
            counts += used
            start += first
        initial = start / start.sum()
        transition = normalized_rows(counts)
    return dict(initial=initial.tolist(), transition=transition.tolist(),
                optical_frames=sum(sum(row is not None for row in point) for point in records))


class DynamicsMemory:
    """Update the regularized transition and opening priors after each fall."""

    def __init__(self, donor=None):
        if donor is None:
            self.initial_counts = np.ones(len(HYPOTHESES))
            self.transition_counts = BASE_TRANSITIONS.copy()
        else:
            self.initial_counts = len(HYPOTHESES) * np.asarray(donor['initial'])
            self.transition_counts = BASE_TRANSITIONS.sum(axis=1, keepdims=True) * np.asarray(donor['transition'])

    @property
    def probabilities(self):
        return self.initial_counts / self.initial_counts.sum()

    @property
    def transition(self):
        return normalized_rows(self.transition_counts)

    def state(self):
        return self.probabilities.tolist()

    def update(self, evidence):
        transitions, first = expected_counts(evidence, self.probabilities, self.transition)
        self.transition_counts += transitions
        self.initial_counts += first


class ChangingDelay:
    """Infer the active delay from public images and learn its dynamics locally."""

    def __init__(self, memory, start, noise):
        self.memory = memory
        self.belief = memory.probabilities
        self.transition = memory.transition
        self.noise = np.asarray(noise)
        self.offset = np.zeros_like(self.noise)
        self.history = deque([np.asarray(start)] * len(HYPOTHESES), maxlen=len(HYPOTHESES))
        self.previous = None
        self.informative = 0
        self.trajectory = []
        self.emissions = []

    @property
    def delay(self):
        return float(self.belief @ HYPOTHESES)

    def advance(self, nominal):
        self.history.appendleft(np.asarray(nominal))
        self.belief = self.belief @ self.transition

    def observe(self, sighting):
        predicted = np.asarray(self.history)
        likelihood = optical_likelihood(sighting, self.previous, predicted, self.offset, self.noise)
        if likelihood is not None:
            self.belief *= likelihood
            self.belief /= self.belief.sum()
            self.informative += 1
        if sighting is not None:
            sighting = np.asarray(sighting)
            bias = sighting - self.belief @ predicted
            self.offset += .035 * np.clip(bias - self.offset, -self.noise, self.noise)
            self.previous = sighting, predicted
        else:
            self.previous = None
        self.trajectory.append(self.delay)
        self.emissions.append(likelihood if likelihood is not None else np.ones(len(HYPOTHESES)))

    def finish(self):
        if self.informative >= 3:
            self.memory.update(self.emissions)
