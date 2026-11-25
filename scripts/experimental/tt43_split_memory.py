"""Keep every reward available during a rally and consolidate distinct successful corrections between rallies."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write


def distinctive_index(earned, stored):
    """Select the state and motor segment least represented in proximal experience."""
    if not stored:
        return len(earned) // 2
    states = np.stack([r[0] for r in stored])
    goals = np.stack([r[1][[3, 8, 14], :5] for r in stored])
    novelty = []
    for state, sequence, _ in earned:
        distances = np.linalg.norm(states - state, axis=1)
        take = np.argsort(distances, kind="stable")[:4]
        motor_gap = np.linalg.norm(goals[take] - sequence[[3, 8, 14], :5], axis=(1, 2))
        novelty.append(np.min(distances[take] + 0.7 * motor_gap))
    return int(np.argmax(novelty))


class SplitMemory(course.Carried):
    def start(self):
        self.stable = list(self.online)
        self.earned = []
        # Every movement rewarded in this point stays available through its end.
        self.online = deque(self.stable)

    def keep(self, archive, leg):
        before = len(self.online)
        n = super().keep(archive, leg)
        if n:
            self.earned.append(list(self.online)[before:])
        return n

    def consolidate(self):
        permanent = deque(self.stable, maxlen=768)
        for earned in self.earned:
            permanent.append(earned[distinctive_index(earned, permanent)])
        self.online = permanent
        return {
            "rewarded_segments": sum(map(len, self.earned)),
            "consolidated": len(self.earned),
        }


def run(a):
    native = course.mpc.play

    def separately(*args, **kwargs):
        model = args[10]
        model.priors.start()
        row = native(*args, **kwargs)
        row["consolidation"] = model.priors.consolidate()
        row["memory"]["after"] = model.priors.digest()
        return row

    with (
        patch.object(course, "Carried", SplitMemory),
        patch.object(course.mpc, "play", separately),
    ):
        course.run(a)
    d = _read(a.out)
    d["memory_rule"] = (
        "All scored movements stay available during each point. One locally distinct joint movement per scored ball is stored between points. Capacity is 768 persistent fragments."
    )
    d["memory_rule_sha256"] = _hash(Path(__file__))
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(mode="live", deck="practice", cap=240, resume=False, carry=None)
    run(p.parse_args())
