"""Rerank safe model rollouts using nearby observed return successes and misses."""

from collections import deque

import numpy as np


class OutcomeMemory:
    """Keep bounded positive and negative evidence from the referee's legal-return signal."""

    def __init__(self):
        self.samples = {True: deque(maxlen=96), False: deque(maxlen=96)}

    def evidence(self, features, success):
        examples = self.samples[success]
        if not examples:
            return np.zeros(len(features))
        distance = np.sum((features[:, None, :] - np.asarray(examples)) ** 2, axis=2)
        return np.exp(-0.5 * distance.min(axis=1))

    def penalty(self, features):
        if not self.samples[False]:
            return np.zeros(len(features))
        failed = self.evidence(features, False)
        passed = self.evidence(features, True)
        return 4.0 * np.maximum(0.0, failed - passed)


class OutcomeFeedback:
    """Associate the executed tape with its simulated stroke, then score the real outcome."""

    # Incoming xyz and velocity, predicted landing xy, net margin, departure speed and rim distance.
    SCALE = np.array([.13, .065, .11, .80, 1.05, 1.15, .34, .43, .125, 1.05, .027])

    def __init__(self, planner, memory, *, learn):
        self.planner = planner
        self.solver = planner.predict
        self.memory = memory
        self.learn = learn
        self.batches = []
        self.last_tape = None
        self.pending = None
        self.received = False
        self.contacted = False
        self.log = []
        self.changed = 0

    def trajectories(self, goals, target, cutoff):
        rates, scores = self.solver.trajectories(goals, target, cutoff)
        valid = np.flatnonzero(np.isfinite(scores[:, 0]))
        if not valid.size:
            return rates, scores
        arena = self.planner.arena
        path = self.planner.flight.path(
            arena.ball, arena.velocity, arena.spin, 1, arena._received_bounce
        )
        if not len(path):
            return rates, scores
        selected = scores[valid]
        incoming = np.stack(
            [np.interp(selected[:, 1], path[:, 0], column) for column in path[:, 1:7].T],
            axis=1,
        )
        features = np.column_stack((incoming, selected[:, [3, 4, 5, 11, 12]])) / self.SCALE
        penalty = self.memory.penalty(features)
        if self.learn:
            adjusted = selected[:, 0] + penalty
            self.changed += int(adjusted.argmin() != selected[:, 0].argmin())
            scores = scores.copy()
            scores[valid, 0] = adjusted
        self.batches.append((rates, valid, features, penalty))
        return rates, scores

    def _selected(self, tape):
        for rates, valid, features, penalty in reversed(self.batches):
            if np.shares_memory(rates, tape):
                row = (tape.ctypes.data - rates.ctypes.data) // rates.strides[0]
                index = np.searchsorted(valid, row)
                if index < len(valid) and valid[index] == row:
                    return features[index], float(penalty[index])
        return None

    def _record(self, event, success):
        if self.pending is not None:
            features, penalty = self.pending
            self.memory.samples[success].append(features)
            self.log.append(
                dict(
                    step=event["step"],
                    success=success,
                    contact=self.contacted,
                    reason=event.get("reason"),
                    penalty=penalty,
                )
            )
        self.pending = None

    def observe(self, tape, events):
        if tape is not self.last_tape:
            if tape is not None and not self.contacted:
                self.pending = self._selected(tape)
            self.last_tape = tape
        self.batches.clear()
        for event in events:
            kind = event["kind"]
            if kind == "contact" and event["side"] == 1:
                self.contacted = True
            elif kind == "legal_return":
                if event["side"] == -1:
                    self.received = True
                else:
                    self._record(event, True)
                    self.received = self.contacted = False
            elif (
                kind == "point"
                and event["winner"] == -1
                and (self.received or self.contacted)
            ):
                self._record(event, False)
