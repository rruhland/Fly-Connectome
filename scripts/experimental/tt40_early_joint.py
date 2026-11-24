"""Use early visual flight to choose a longer preparation and direct joint movement."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_carry_joints import Carried
from .tt39_turn_time import turn_time

SHORT = mpc.HORIZON
LONG = 39


class Extended(Carried):
    def __init__(self, bank, engine, **kwargs):
        with patch.object(mpc, "HORIZON", LONG):
            super().__init__(bank, engine, **kwargs)

    def priors(self, x, q, reference, limit):
        return [
            (mpc.tail(seq, mpc.HORIZON), earned)
            for seq, earned in super().priors(x, q, reference, limit)
        ]

    def keep(self, archive, leg):
        hit = turn_time([(t, x) for t, x, *_ in archive])
        if hit is None:
            return 0
        ids = [
            i
            for i, (t, x, *_) in enumerate(archive)
            if hit - LONG <= t <= hit - 3 and x is not None and x[4] < -0.9
        ]
        for i in ids[::3]:
            _, x, q, ref, _ = archive[i]
            moves = np.asarray([r[-1] for r in archive[i : i + LONG]])
            self.online.append((mpc.query(x, q, ref), mpc.tail(moves, LONG), leg))
        return len(ids[::3])


class Early(mpc.MPC):
    def begin(self):
        super().begin()
        self.prepared = False

    def step(self, tick, eye, x, q, reference, base):
        self.engine.observe(tick, eye, q, reference.copy())
        if x is not None and x[4] > 0.9:
            self.best = None
            self.prepared = False
            return base
        if x is not None and x[4] < -0.9 and 0.38 > x[1] > -0.47 and not self.prepared:
            # A fresh optical observation after the far hit allows an early
            # change of the body configuration. Uncertain attempts leave the
            # pretrained camera-to-joint policy in control.
            self.prepared = True
            with patch.object(mpc, "HORIZON", LONG):
                self.search(tick, x, q, reference, base)
            if self.best is not None and self.notes[-1].get("robust", 0) >= 1:
                self.next = tick + 12
                self.lease = 12
                self.diag["early_plans"] += 1
            else:
                self.best = None
        if (
            x is not None
            and x[4] < -0.9
            and -0.49 > x[1] > -1.62
            and x[2] > 0.77
            and tick >= self.next
        ):
            if self.best is not None and len(self.best) > SHORT:
                self.best = mpc.tail(self.best[max(0, tick - self.tick) :], SHORT)
                self.tick = tick
            self.next = tick + 5
            self.search(tick, x, q, reference, base)
        if self.best is not None and 0 <= tick - self.tick < self.lease:
            self.diag["screened_ticks"] += 1
            return self.best[tick - self.tick]
        return base


def run(a):
    held = {r["id"] for r in _read(a.plan)[a.deck][: a.limit]}
    objects = []

    def model(bank, engine):
        obj = Extended(bank, engine, practice=a.practice, held=held)
        objects.append(obj)
        return obj

    with patch.object(mpc, "Fragments", model), patch.object(mpc, "MPC", Early):
        mpc.run(
            a.feeds,
            mpc.BANK,
            a.plan,
            a.deck,
            "live",
            a.out,
            optic=mpc.ROOT / "tt37/rebound.npz",
            limit=a.limit,
            variant="elastic",
            cap=240,
        )
    tape = a.out.with_suffix("").with_suffix(".npz")
    rows = objects[0].online
    np.savez_compressed(
        tape,
        states=[r[0] for r in rows],
        goals=[r[1] for r in rows],
        tags=[repr(r[2]) for r in rows],
    )
    doc = _read(a.out)
    doc.update(
        previous_practice={str(p): _hash(p) for p in a.practice},
        memory_rule="Prior scored joint sequences, then own score-gated updates",
        checkpoint_sha256=_hash(tape),
        preparation_horizon=LONG,
    )
    _write(a.out, doc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--plan", type=Path, default=mpc.ROOT / "tt37/memory-course.json"
    )
    parser.add_argument("--deck", default="replication")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--practice",
        nargs="*",
        type=Path,
        default=[mpc.ROOT / "tt37/camera-time-school.npz"],
    )
    parser.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    run(parser.parse_args())
