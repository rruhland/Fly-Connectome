"""Assign episodic object tracks using the unchanged native Pong role model."""

from collections import deque

import numpy as np

from .tt3_visual import RoleBinder, Track
from .tt22_relation import Tracks


class CandidateFrames:
    """Convert anonymous appearance regions to the source model's track interface."""

    def __init__(self, library):
        self.tracker = Tracks(library)
        self._identities = {}
        self._next = 0
        self.current = ()

    def observe(self, frame):
        self.tracker.update(frame)
        tick = self.tracker.tick
        active = {}
        candidates = []
        for region in self.tracker.tracks:
            key = id(region)
            identity, age = self._identities.get(key, (None, 0))
            if identity is None:
                identity = self._next
                self._next += 1
            seen = region.history[-1][0] == tick
            age += seen
            active[key] = (identity, age)
            if seen:
                candidates.append(Track(identity, tuple(region.position), tuple(region.velocity()), True, age))
        self._identities = active
        self.current = tuple(candidates)
        return self.current

    def update(self, frame):
        # One immutable candidate set is shared by the actual and control rankers.
        return self.current


class AnonymousEye:
    """Deployable single-eye binder. Prior commands are the only non-image input."""

    def __init__(self, library, role_model):
        self.candidates = CandidateFrames(library)
        self.binder = self.compare(role_model)

    def compare(self, role_model):
        binder = RoleBinder(role_model)
        binder._scene.tracker = self.candidates
        return binder

    def update(self, frame, previous_command):
        self.candidates.observe(frame)
        return self.binder.update(frame, previous_command)


def pixel(track, image_shape):
    height, width = image_shape[:2]
    return np.array((width, height)) / 2. + np.asarray(track.position) * max(width, height)


class Controls:
    def __init__(self, eye, shuffled, seed):
        self.reward_shuffled = eye.compare(shuffled)
        self.command_shuffled = eye.compare(eye.binder.model)
        self.rng = np.random.default_rng(seed)
        self.past = deque(maxlen=25)

    def update(self, frame, command, primary):
        sham = None
        if command is not None:
            if self.past:
                sham = self.past[self.rng.integers(len(self.past))]
            self.past.append(command)
        reward_shuffled = self.reward_shuffled.update(frame, command)
        command_shuffled = self.command_shuffled.update(frame, sham)
        eligible = [t for t in primary.tracks if t.age >= 5]
        speed = sorted(eligible, key=lambda track: np.linalg.norm(track.velocity), reverse=True)
        movers = [track for track in speed if primary.controlled is None or track.id != primary.controlled.id]
        random = self.rng.choice(eligible, size=2, replace=False) if len(eligible) >= 2 else (None, None)
        return {
            'pong': (primary.controlled, primary.relevant),
            'reward_shuffled': (reward_shuffled.controlled, reward_shuffled.relevant),
            'command_shuffled': (command_shuffled.controlled, command_shuffled.relevant),
            'speed': (None, movers[0] if movers else None),
            'random': tuple(random),
        }
