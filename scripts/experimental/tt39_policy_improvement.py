"""Protect a competent joint policy's proposals while learning extra movements from scores and misses."""

import argparse
from collections import deque
from pathlib import Path
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_joint_credit import Credited


class Conservative(mpc.MPC):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.failures = deque(maxlen=450)

    def begin(self):
        super().begin()
        self.stock = None

    def missed(self, archive):
        inward = [
            (i, r)
            for i, r in enumerate(archive)
            if r[1] is not None and r[1][4] < -0.9 and r[1][1] < -0.85
        ]
        if not inward:
            return
        # Observed ball and the arm's forward body map locate the nearest instant
        # on a scoreless attempt, even for a complete whiff.
        latest = inward[-1][1][0]
        candidates = [(i, r) for i, r in inward if latest - r[0] < 18]
        nearest = min(
            candidates,
            key=lambda a: np.linalg.norm(
                self.geometry.pose(a[1][2][:5])[0] - a[1][1][:3]
            ),
        )[1][0]
        for i, (t, x, q, prev, _) in inward[::3]:
            if not nearest - 20 <= t <= nearest - 3:
                continue
            u = mpc.tail(
                np.asarray([r[-1] for r in archive[i : i + mpc.HORIZON]]), mpc.HORIZON
            )
            self.failures.append((mpc.query(x, q, prev), u))
        self.diag["stored_failed_fragments"] = len(self.failures)

    def search(self, tick, x, q, reference, base):
        serial = self.priors.online
        self.priors.online = ()
        if self.stock is not None:
            self.best, self.tick, self.lease = self.stock
        try:
            super().search(tick, x, q, reference, base)
        finally:
            self.priors.online = serial
        self.stock = self.best, self.tick, self.lease
        if not serial or not self.notes or self.notes[-1]["tick"] != tick:
            return
        timer = process_time()
        note = self.notes[-1]
        if note.get("robust") == 2 and note.get("score", 10) < -1.7:
            return
        key = mpc.query(x, q, reference)
        chosen = []
        used = set()
        st = np.stack([r[0] for r in serial])
        distances = np.linalg.norm(st - key, axis=1)
        for i in np.argsort(distances):
            _, seq, tag = serial[i]
            if distances[i] > 3.0 or len(chosen) >= 4:
                break
            if tag not in used:
                chosen.append(seq)
                used.add(tag)
        if not chosen:
            return
        # Identical stock proposals, search RNG, and acceptance always run first.
        # New actions cannot evict a stock sample from that candidate budget.
        warm = (
            mpc.tail(self.best[tick - self.tick :], mpc.HORIZON)
            if self.best is not None and 0 <= tick - self.tick < self.lease
            else np.broadcast_to(base, (mpc.HORIZON, 6)).copy()
        )
        options = [warm, *chosen]
        _, _, tr, _, _ = self.trial(chosen, traces=True)
        for seq, trace in zip(chosen, tr, strict=True):
            delta = self.transport(trace, face=True)
            if delta is not None:
                options.append(seq + 0.78 * delta)
        loss, legal, _, y, _ = self.trial(options, long=True)
        acceptable = np.flatnonzero(legal)
        if len(acceptable) and np.any(acceptable > 0):
            opts = [options[i] for i in acceptable]
            support, quality = self.uncertainty(opts, loss[acceptable])
            risk = quality + 1.85 * (2 - support)
            if self.failures:
                ds = np.linalg.norm(
                    np.stack([r[0] for r in self.failures]) - key, axis=1
                )
                idx = np.argsort(ds)[:6]
                for j, opt in enumerate(opts):
                    if acceptable[j] == 0:
                        continue
                    gap = np.array(
                        [
                            np.mean(
                                ((opt[:8, :5] - self.failures[i][1][:8, :5]) / 0.12)
                                ** 2
                            )
                            for i in idx
                        ]
                    )
                    risk[j] += 1.1 * float(
                        np.max(np.exp(-0.5 * ds[idx] ** 2 - 0.5 * gap))
                    )
            j = int(np.argmin(risk))
            i = acceptable[j]
            prev = np.flatnonzero(acceptable == 0)
            enough = not len(prev) or risk[j] + 0.48 < risk[prev[0]]
            if i > 0 and enough and support[j] >= 1:
                self.best = options[i]
                self.tick = tick
                self.lease = mpc.LEASE
                self.diag["separate_memory_wins"] += 1
                note.update(
                    chosen="additional_rewarded_motion",
                    robust=int(support[j]),
                    score=float(risk[j]),
                    previously_viable=bool(len(prev)),
                    memory_contact_radius=float(y[i, 12]),
                )
        self.diag["search_cpu_s"] += process_time() - timer


def run(a):
    with (
        patch.object(mpc, "MPC", Conservative),
        patch.object(mpc, "Fragments", Credited),
    ):
        mpc.run(
            a.feeds,
            mpc.BANK,
            a.plan,
            a.deck,
            "live",
            a.output,
            optic=mpc.ROOT / "tt37/rebound.npz",
            limit=a.limit,
            variant="elastic",
            cap=200,
        )
    d = _read(a.output)
    d["extra_actor"] = _hash(Path(__file__))
    _write(a.output, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deck")
    p.add_argument("output", type=Path)
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--plan", type=Path, default=mpc.PLAN)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    run(p.parse_args())
