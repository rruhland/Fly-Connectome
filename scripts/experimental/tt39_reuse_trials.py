"""Reuse exact private-model trajectories while comparing overlapping actuator proposals."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write


class Cached(mpc.MPC):
    def search(self, *args):
        self.tables = {}
        try:
            return super().search(*args)
        finally:
            self.tables = {}

    def trial(self, goals, *, long=False, traces=False):
        a = self.engine.twin
        # A search performs one optical synchronization. Only hypothetical translation,
        # speed and spin change thereafter, and each of those hypotheses has its own table.
        initial = (
            a.data.qpos.tobytes(),
            a.data.qvel.tobytes(),
            a.data.ctrl.tobytes(),
            bool(a._received_bounce),
            long,
        )
        table = self.tables.setdefault(initial, {})
        frames = mpc.HORIZON + (mpc.TAIL if long else 0)
        keys = [mpc.tail(seq, frames).tobytes() for seq in goals]
        needed = {}
        for key, seq in zip(keys, goals, strict=True):
            if key not in table or (traces and table[key][2] is None):
                needed.setdefault(key, seq)
        if needed:
            loss, valid, track, scores, response = super().trial(
                list(needed.values()), long=long, traces=traces
            )
            for i, key in enumerate(needed):
                table[key] = (
                    loss[i],
                    valid[i],
                    None if track is None else track[i],
                    scores[i],
                    response[i],
                )
        self.diag["exact_trials_reused"] += len(keys) - len(needed)
        rows = [table[k] for k in keys]
        return (
            np.array([r[0] for r in rows]),
            np.array([r[1] for r in rows]),
            np.stack([r[2] for r in rows]) if traces else None,
            np.stack([r[3] for r in rows]),
            np.stack([r[4] for r in rows]),
        )


def run(a):
    with patch.object(mpc, "MPC", Cached):
        mpc.run(
            a.feeds,
            mpc.BANK,
            mpc.PLAN,
            a.deck,
            "live",
            a.output,
            optic=mpc.ROOT / "tt37/rebound.npz",
            limit=a.limit,
            variant="elastic",
            cap=200,
        )
    d = _read(a.output)
    d["cache_code"] = _hash(Path(__file__))
    _write(a.output, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--deck", default="held_openings")
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    run(p.parse_args())
