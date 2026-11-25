"""Observe each point in full, then preserve a bounded sample of new scored motor experience."""

import argparse
from collections import deque
from hashlib import blake2b
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt46_continuous_season as season
from . import tt49_cumulative_memory as history
from . import tt52_consolidation as prior
from . import tt52_practice_dose as dose
from .tt4_transfer_trial import _hash, _read, _write
from .tt54_training_speedups import enabled

ROOT = Path("artifacts/tt54_stratified_memory")
course = season.course


class MotorSample(history.Growing):
    def restore_banks(self, parent, extra):
        self.parent = parent
        self.states_parent = np.array([r[0] for r in parent])
        self.motor_parent = np.array([prior.motor(r) for r in parent])
        self.ranks = {}
        novelty = [self.score(r)[0] for r in extra]
        self.breaks = np.quantile(novelty, [0.25, 0.5, 0.75])
        self.online = deque([*parent, *extra])
        self.old_size = len(self.online)

    def score(self, r):
        h = blake2b(digest_size=16)
        h.update(np.ascontiguousarray(r[0], dtype="<f8").tobytes())
        h.update(np.ascontiguousarray(r[1], dtype="<f8").tobytes())
        key = h.digest()
        if key not in self.ranks:
            state_dist = np.mean((self.states_parent - r[0]) ** 2, axis=1)
            motor_dist = np.mean((self.motor_parent - prior.motor(r)) ** 2, axis=1)
            self.ranks[key] = float(np.sqrt(np.min(state_dist + 0.65**2 * motor_dist)))
        return self.ranks[key], key

    def settle(self):
        if self.old_size == len(self.online):
            return 0
        pools = [{} for _ in range(4)]
        for r in list(self.online)[len(self.parent) :]:
            distance, priority = self.score(r)
            pools[int(np.searchsorted(self.breaks, distance))].setdefault(priority, r)
        kept = [pool[k] for pool in pools for k in sorted(pool)[:48]]
        dropped = len(self.online) - len(self.parent) - len(kept)
        self.online = deque([*self.parent, *kept])
        self.ranks = {k: self.ranks[k] for pool in pools for k in sorted(pool)[:48]}
        self.stamp = None
        self.old_size = len(self.online)
        return dropped


def run(a):
    ROOT.mkdir(exist_ok=True, parents=True)
    parent = history.load_rows(season.CARRY)
    allrows, _ = history.restore_history(24)
    with np.load(prior.MODEL) as z:
        extra = [allrows[int(j)] for j in z["matched"]]
    a.carry = season.ROOT / "learning-24.json.gz"
    used_rows = [*_read(season.CARRY)["cards"][:48], *_read(a.carry)["cards"][:24]]
    tested = _read(dose.PLAN)["practice"][: a.n]
    if {c["id"] for c in used_rows} & {c["id"] for c in tested}:
        raise ValueError("Earlier practice enters transfer evaluation")
    impl = course.mpc.play
    rank = season.fresh.shot.Coverage(_read(season.fresh.RISK))

    def restore(actor, p, d):
        prior.retention.ancestral_restore(actor, p, d, season.CARRY)
        actor.priors.restore_banks(parent, extra)

    def play(*args):
        r = impl(*args)
        bank = args[10].priors
        pre = r["memory"]["after"]
        dropped = bank.settle()
        r["memory"].update(
            before_consolidation=pre, after=bank.digest(), fragments_removed=dropped
        )
        return r

    with (
        patch.object(course, "PLAN", dose.PLAN),
        patch.object(course, "Carried", MotorSample),
        patch.object(course, "resume", restore),
        patch.object(course.mpc, "play", play),
        patch.object(
            course.mpc, "MPC", lambda *x, **kw: season.Player(*x, coverage=rank, **kw)
        ),
        enabled(),
    ):
        course.run(a)
    d = _read(a.out)
    d["earlier_practice"] = [_hash(season.CARRY), _hash(a.carry), _hash(prior.MODEL)]
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        mode="live",
        deck="practice",
        cap=240,
        resume=False,
        out=ROOT / "continuous.json.gz",
    )
    run(p.parse_args())
