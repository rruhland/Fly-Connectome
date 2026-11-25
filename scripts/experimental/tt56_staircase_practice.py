"""Revisit a recent miss near a scoreable practice shot, with continuous local exploration."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt53_responsive_school as baseline
from .tt54_training_speedups import enabled

School = baseline.School
ROOT = Path("artifacts/tt56_staircase_practice")


class Staircase(School):
    def __init__(self, adaptive, budget, earlier):
        self.successful = deque(maxlen=384)
        self.missed = deque(maxlen=48)
        self.trials = []
        self.active = None
        # During previous practice the teacher can observe the feeds it delivered and
        # whether the arm's response legally landed. The actor is unaware of this record.
        self.acquiring = False
        super().__init__(adaptive, budget, earlier)
        self.acquiring = True
        self.schedule = np.random.default_rng(562341)

    def observe(self, row):
        super().observe(row)
        for ball in row["placements"]["delivered"]:
            if not ball.get("near_bounce"):
                continue
            goal = np.clip(
                (ball["target"] - baseline.BOUNDS[:, 0])
                / np.ptp(baseline.BOUNDS, axis=1),
                0,
                1,
            )
            if ball["near_landed"]:
                self.successful.append(goal)
            else:
                self.missed.append(goal)
            if self.acquiring:
                self.trials.append((goal, bool(ball["near_landed"])))
        if self.acquiring and self.active is not None:
            recent = self.trials[self.active["start"] :]
            local = [
                y for g, y in recent if np.linalg.norm(g - self.active["aim"]) < 0.33
            ]
            if local:
                # Advance on an earned return, retreat after repeated local mistakes.
                self.active["fraction"] = np.clip(
                    self.active["fraction"]
                    + (0.16 if np.mean(local) >= 0.65 else -0.12),
                    0.12,
                    1.0,
                )
                self.active["sessions"] += 1
            if self.active["sessions"] >= 4 or self.active["fraction"] >= 0.98:
                self.active = None

    def choose_goal(self):
        misses = np.asarray(self.missed)
        successes = np.asarray(self.successful)
        if not len(misses) or not len(successes):
            return None
        # Find a failed feed with physically successful neighbors. Neighbors may
        # yield different actual replies after different outgoing strokes.
        sample = self.schedule.choice(len(misses), min(8, len(misses)), replace=False)
        distances = np.linalg.norm(misses[sample, None] - successes[None], axis=2)
        neighbors = np.argsort(distances, axis=1)
        closest = distances[
            np.arange(len(sample)), neighbors[:, min(3, len(successes) - 1)]
        ]
        i = int(np.argmin(abs(closest - 0.32)))
        ok = int(neighbors[i, self.schedule.integers(min(4, len(successes)))])
        return dict(
            good=successes[ok],
            bad=misses[sample[i]],
            fraction=0.55,
            sessions=0,
            aim=successes[ok],
            start=len(self.trials),
        )

    def draw(self, card, leg, protocol, rng):
        key = (card["seed"], card["reply_seed"], leg)
        if key in self.draws:
            return self.draws[key]
        original, name = baseline.DRAW(card, leg, protocol, rng)
        if not self.acquiring:
            return original, name
        if self.active is None:
            self.active = self.choose_goal()
        if self.active is None:
            return original, name
        state = self.active
        if leg == 0:
            state["start"] = len(self.trials)
        center = (
            state["good"] * (1 - state["fraction"]) + state["bad"] * state["fraction"]
        )
        state["aim"] = center
        random = np.random.default_rng([*key, 56721])
        if random.uniform() < 0.45:
            output = original, "staircase_uniform"
        else:
            # The truncated local box stays continuous in all coordinates, even
            # where a practiced success or miss was near the edge of the court.
            radius = 0.12 + 0.13 * state["fraction"]
            lo, hi = np.maximum(0, center - radius), np.minimum(1, center + radius)
            u = (np.asarray(original) - baseline.BOUNDS[:, 0]) / np.ptp(
                baseline.BOUNDS, axis=1
            )
            goal = baseline.BOUNDS[:, 0] + (lo + u * (hi - lo)) * np.ptp(
                baseline.BOUNDS, axis=1
            )
            output = tuple(goal), "staircase_local"
        self.draws[key] = output
        return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("live", "frozen", "transfer"))
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--budget", type=int, default=192)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(kind="staircase", resume=False, cap=240)
    with (
        enabled(),
        patch.object(baseline, "School", Staircase),
        patch.object(baseline, "ROOT", ROOT),
    ):
        baseline.run(p.parse_args())
