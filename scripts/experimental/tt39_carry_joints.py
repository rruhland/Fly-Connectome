"""Carry a sparse library of optically credited, executed joint segments to new games."""

import argparse
import itertools
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_joint_credit import Credited


class Carried(Credited):
    def __init__(self, bank, engine, *, practice, held, late=False):
        super().__init__(bank, engine)
        self.late = late
        self.online = deque(maxlen=768)
        for file in practice:
            receipt = _read(file.with_suffix(".json"))
            if any(c["card"]["id"] in held for c in receipt["practice"]):
                raise ValueError("Pretraining includes a held release")
            original = BASE(file, engine)
            by_motion = {}
            for state, commands, (i, stamp) in original.rows:
                by_motion.setdefault((i, stamp), []).append((state, commands))
            for (i, stamp), points in by_motion.items():
                for state, commands in points[::4]:
                    self.online.append(
                        (state, commands, ("carried", file.stem, i, stamp))
                    )

    def keep(self, archive, leg):
        if not self.late:
            return super().keep(archive, leg)
        # Same density and eligibility as fitted credit. Only the estimated
        # time of contact is changed for this control.
        visible = [r for r in archive if r[1] is not None]
        candidates = [
            (a[0] + b[0]) * 0.5
            for a, b in itertools.pairwise(visible)
            if a[1][4] < -0.9 and b[1][4] > 0.9 and b[0] - a[0] <= 9
        ]
        if not candidates:
            return 0
        hit = candidates[-1]
        indices = [
            i
            for i, (t, x, *_) in enumerate(archive)
            if hit - 21 <= t <= hit - 3
            and x is not None
            and x[4] < -0.9
            and x[1] < -0.62
        ]
        for i in indices[::3]:
            _, x, q, prev, _ = archive[i]
            seq = mpc.tail(
                np.asarray([r[-1] for r in archive[i : i + mpc.HORIZON]]), mpc.HORIZON
            )
            self.online.append((mpc.query(x, q, prev), seq, leg))
        return len(indices[::3])


BASE = mpc.Fragments


def run(args):
    roster = _read(args.plan)[args.deck][: args.limit]
    instances = []

    def factory(bank, engine):
        obj = Carried(
            bank,
            engine,
            practice=args.practice,
            held={r["id"] for r in roster},
            late=args.late,
        )
        if args.capacity is not None or not args.practice:
            obj.online = deque(obj.online, maxlen=args.capacity or 192)
        instances.append(obj)
        return obj

    with patch.object(mpc, "Fragments", factory):
        mpc.run(
            args.feeds,
            mpc.BANK,
            args.plan,
            args.deck,
            "live",
            args.out,
            optic=mpc.ROOT / "tt37/rebound.npz",
            limit=args.limit,
            cap=200,
            variant="elastic",
        )
    model = instances[0]
    tape = args.out.with_suffix("").with_suffix(".npz")
    np.savez_compressed(
        tape,
        states=np.asarray([r[0] for r in model.online]),
        goals=np.asarray([r[1] for r in model.online]),
        tags=np.array([repr(r[2]) for r in model.online]),
    )
    report = _read(args.out)
    report.update(
        credit_method="observed reversal"
        if args.late
        else "intersected flight segments",
        previous_practice={f.name: _hash(f) for f in args.practice},
        online_boundary="Carried scored practice followed by own score-gated updates"
        if args.practice
        else "Empty adaptive memory followed by own score-gated updates",
        capacity=model.online.maxlen,
        checkpoint_sha256=_hash(tape),
    )
    _write(args.out, report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deck")
    p.add_argument("out", type=Path)
    p.add_argument("--plan", type=Path, default=mpc.PLAN)
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--late", action="store_true")
    p.add_argument("--capacity", type=int)
    p.add_argument("--practice", type=Path, nargs="*", default=[])
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    run(p.parse_args())
