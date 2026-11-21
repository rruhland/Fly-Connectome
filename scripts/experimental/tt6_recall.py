"""Identify a returning motor interface from a short intervention during the serve."""
from dataclasses import replace
from itertools import permutations, product

import numpy as np

from .tt5_player_body import ANGLE_HIGH, ANGLE_LOW, BodyServo, centers


class InterfaceMemory:
    """Compare rendered blade motion to signed permutations of a visually fitted arm."""

    def __init__(self, fitted, *, remember=True, horizon=24, early=True):
        self.fitted = fitted
        self.horizon = horizon
        self.remember = remember
        self.early = early
        pairs = [(order, signs) for order in permutations(range(4))
                 for signs in product((-1, 1), repeat=4)]
        self.order, self.sign = map(np.array, zip(*pairs, strict=True))
        self.gain = self.sign * np.abs(fitted.gains)
        self.base = pairs.index((tuple(fitted.scripts), tuple(np.sign(fitted.gains).astype(int))))
        # Both agents start with the installation on which the body was first fitted.
        self.seen = {self.base: 1}
        # A bounded, zero-mean sequence moves the blade in distinguishable directions.
        directions = np.array([[.28, .55, -.75, .94], [.85, -.31, .54, .22],
                               [-.43, .89, .23, -.68]])
        self.tape = np.repeat(np.concatenate([np.stack((d, -d)) for d in directions]), 4, axis=0)
        if not (12 <= horizon <= len(self.tape)):
            raise ValueError("intervention horizon must be between 12 and 24 commands")

    def reset(self):
        self.requested = np.broadcast_to(self.fitted.home, self.gain.shape).copy()
        self.effective = self.requested.copy()
        self.origin = None
        self.origin_pitch = None
        self.loss = np.zeros(len(self.order))
        self.observations = 0
        self.selected = None
        self.elapsed = 0

    def update(self, preceding, actor, pitch):
        if preceding is not None:
            self.elapsed += 1
            self.requested = np.clip(self.requested + self.gain * preceding[self.order], ANGLE_LOW, ANGLE_HIGH)
            self.effective += (1. - self.fitted.lags) * (self.requested - self.effective)
        if actor is not None and actor.observed:
            measured = np.array(actor.position)
            if self.origin is None:
                self.origin = measured
                self.start_states = self.effective.copy()
            predicted = centers(self.effective, self.fitted.geometry) - centers(self.start_states, self.fitted.geometry)
            discrepancy = (predicted - (measured - self.origin)) / [.012, .045, .017]
            self.loss += np.minimum(np.abs(discrepancy), 6.).sum(axis=1)
            if pitch is not None:
                if self.origin_pitch is None:
                    self.origin_pitch = pitch
                    self.start_pitch = self.effective[:, 3].copy()
                error = self.effective[:, 3] - self.start_pitch - (pitch - self.origin_pitch)
                self.loss += np.minimum(np.abs(error) / .09, 6.)
            self.observations += 1
        best = int(self.loss.argmin())
        if self.elapsed >= self.horizon:
            self.selected = best
        elif self.early and self.elapsed in (8, 12, 16, 20) and self.observations >= .75 * self.elapsed:
            stored = np.array(sorted(self.seen))
            ranked = stored[np.argsort(self.loss[stored])]
            candidate = int(ranked[0])
            gap = self.loss[candidate] - self.loss[best]
            if (gap <= .14 * self.observations and
                    (len(ranked) == 1 or self.loss[ranked[1]] - self.loss[candidate] > .24 * self.observations)):
                self.selected = candidate
            else:
                two = np.partition(self.loss, 1)[:2]
                separation = (two[1] - two[0]) / self.observations
                if separation > (1.2 if self.elapsed == 8 else .65):
                    self.selected = best
        return self.selected

    def adapter(self):
        return replace(self.fitted, scripts=self.order[self.selected], gains=self.gain[self.selected])

    def store(self):
        if self.remember and self.selected is not None and self.observations >= 8:
            self.seen[self.selected] = self.seen.get(self.selected, 0) + 1


class RecallPlayer:
    """Keep the ballistic player running during in-point identification and reconstruct its servo."""

    def __init__(self, player, memory):
        self.player = player
        self.memory = memory

    def reset(self):
        self.memory.reset()
        self.player.reset()
        self.history = []

    def update(self, frames, preceding=None):
        command = self.player.update(frames, preceding)
        if self.memory.selected is not None:
            return command
        tracker = self.player.scene.tracker
        actor, pitch = tracker.actor, tracker.actor_pitch
        self.history.append((preceding, actor, pitch))
        if self.memory.update(preceding, actor, pitch) is None:
            return self.memory.tape[self.memory.elapsed]
        self.player.adapter = self.memory.adapter()
        servo = BodyServo(self.player.adapter)
        for action, observation, angle in self.history:
            servo.advance(action)
            servo.observe(observation, angle)
        self.player.servo = servo
        return np.zeros(4)

    def finish(self, reward):
        memory = self.memory
        previously_seen = memory.selected in memory.seen
        memory.store()
        return dict(**self.player.finish(reward), probe_commands=memory.elapsed,
                    visible_samples=memory.observations, selected=memory.selected,
                    recognized=previously_seen, library_size=len(memory.seen),
                    inferred_order=self.player.adapter.scripts.tolist(),
                    inferred_sign=np.sign(self.player.adapter.gains).astype(int).tolist())
