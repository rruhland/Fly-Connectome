"""Attach direct motor experience to an optically reconstructed contact instead of a delayed tracker reversal."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_turn_time import turn_time


class Credited(mpc.Fragments):
    def keep(self, archive, leg):
        hit = turn_time([(t, x) for t, x, *_ in archive])
        if hit is None:
            return 0
        ids = [
            i
            for i, (t, x, q, prev, sent) in enumerate(archive)
            if hit - 21 <= t <= hit - 3
            and x is not None
            and x[4] < -0.9
            and x[1] < -0.62
        ]
        n = 0
        for i in ids[::3]:
            _, x, q, prev, _ = archive[i]
            goals = np.asarray([r[-1] for r in archive[i : i + mpc.HORIZON]])
            self.online.append(
                (mpc.query(x, q, prev), mpc.tail(goals, mpc.HORIZON), leg)
            )
            n += 1
        return n


def run(a):
    with patch.object(mpc, "Fragments", Credited):
        mpc.run(
            a.feeds,
            mpc.BANK,
            a.plan,
            a.deck,
            "live",
            a.output,
            optic=a.optic,
            limit=a.limit,
            variant="elastic",
            cap=200,
        )
    d = _read(a.output)
    d["optical_credit_timing"] = _hash(Path(__file__))
    _write(a.output, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deck")
    p.add_argument("output", type=Path)
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--plan", type=Path, default=mpc.PLAN)
    p.add_argument("--optic", type=Path, default=mpc.ROOT / "tt37/rebound.npz")
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    run(p.parse_args())
