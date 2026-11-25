"""Attribute real failures to executed motor suggestions and make room for other scored movements."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write

mpc = course.mpc
ROOT = Path("artifacts/tt46_failures")


class MotorCredit(course.Carried):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes = deque(maxlen=500)
        self.pending = []

    def record(self, tick, x, q, ref, plan):
        # Only the first commands of this candidate are about to be issued.
        # The remaining, unexecuted suffix receives no outcome credit.
        self.pending.append(
            (tick, mpc.query(x, q, ref), np.asarray(plan)[:3, :5].copy())
        )

    def hear(self, tape, reward):
        keys = [
            t for t, x, *_ in tape if x is not None and x[4] < -0.9 and x[1] < -0.87
        ]
        if keys:
            last = keys[-1]
            for tick, key, proposal in self.pending:
                if last - 23 <= tick <= last + 2:
                    issued = [r[-1][:5] for r in tape if tick <= r[0] < tick + 3]
                    if len(issued) == 3:
                        self.outcomes.append((key, np.asarray(issued), bool(reward)))
        self.pending.clear()

    def keep(self, tape, leg):
        self.hear(tape, True)
        return super().keep(tape, leg)

    def priors(self, x, q, ref, limit):
        if not self.outcomes or not any(not r[2] for r in self.outcomes):
            return super().priors(x, q, ref, limit)
        original = super().priors(x, q, ref, limit)
        if len(original) < 2:
            return original
        key = mpc.query(x, q, ref)
        contexts = np.array([r[0] for r in self.outcomes])
        dist = np.linalg.norm(contexts - key, axis=1)
        use = np.flatnonzero(dist < 1.70)
        if not any(not self.outcomes[i][2] for i in use):
            return original
        extras = super().priors(x, q, ref, limit + 4)
        stride = self.stride
        top = []
        for j, (seq, tag) in enumerate(extras):
            issued = ref[:5].copy()
            controls = []
            for target in seq[:3, :5]:
                issued = issued + np.clip(target - issued, -stride, stride)
                controls.append(issued.copy())
            motor = np.array([self.outcomes[i][1] for i in use])
            d = np.sqrt(np.mean((motor - controls) ** 2, axis=(1, 2)))
            weight = np.exp(-2.0 * dist[use] ** 2 - 1.0 * (d / 0.078) ** 2)
            losses = np.array([not self.outcomes[i][2] for i in use])
            penalty = (weight @ losses) / (1.0 + weight.sum())
            top.append((j + 6.0 * penalty, j, seq, tag))
        selected = sorted(top, key=lambda r: r[0])[:limit]
        self.penalized += int([r[1] for r in selected] != list(range(len(original))))
        return [(p, g) for _, _, p, g in selected]


class Actor(season.Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.priors.stride = (
            0.8
            * self.engine.twin._joint_increments[self.engine.actuators][:5]
            * self.engine.predict.micro
        )

    def begin(self):
        super().begin()
        self.priors.pending = []
        self.priors.penalized = 0

    def search(self, tick, x, q, reference, base):
        super().search(tick, x, q, reference, base)
        if self.best is not None and self.tick == tick:
            self.priors.record(tick, x, q, reference, self.best)

    def missed(self, tape):
        self.priors.hear(tape, False)


def run(a):
    data = fresh.shot.Coverage(_read(fresh.RISK))
    ordinary = mpc.play

    def play(*args, **kwargs):
        row = ordinary(*args, **kwargs)
        memory = args[10].priors
        row["ranked_from_outcomes"] = memory.penalized
        row["failed_movements_in_memory"] = sum(not r[2] for r in memory.outcomes)
        return row

    models = []

    def actor(*args, **kwargs):
        obj = Actor(*args, coverage=data, **kwargs)
        models.append(obj)
        return obj

    with (
        patch.object(course, "PLAN", fresh.PLAN),
        patch.object(course, "Carried", MotorCredit),
        patch.object(mpc, "MPC", actor),
        patch.object(mpc, "play", play),
    ):
        course.run(a)
    d = _read(a.out)
    d.update(
        source=_hash(Path(__file__)),
        outcomes=[(k.tolist(), v.tolist(), r) for k, v, r in models[0].priors.outcomes],
    )
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--n", default=24, type=int)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        deck="assessment", mode="live", cap=240, carry=fresh.CARRY, resume=False
    )
    a = p.parse_args()
    a.out.parent.mkdir(exist_ok=True, parents=True)
    run(a)
