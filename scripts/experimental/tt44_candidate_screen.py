"""Verify whether the cheap near-contact screen discards already available legal joint movements."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write


class Unscreened(course.mpc.MPC):
    def trial(self, goals, *, long=False, traces=False):
        if getattr(self, "capture", False) and not traces:
            if long:
                self.examined.update(np.asarray(g).tobytes() for g in goals)
            elif len(goals) >= 10:
                self.pool.extend(np.asarray(g) for g in goals)
        return super().trial(goals, long=long, traces=traces)

    def search(self, tick, x, q, reference, base):
        self.examined = set()
        self.pool = []
        self.capture = True
        try:
            super().search(tick, x, q, reference, base)
        finally:
            self.capture = False
        if not self.notes or self.notes[-1]["tick"] != tick or not self.engine.bounced:
            return
        note = self.notes[-1]
        if note.get("robust", 0) > 0:
            return
        skipped = []
        for seq in self.pool:
            key = seq.tobytes()
            if key not in self.examined:
                skipped.append(seq)
                self.examined.add(key)
        if not skipped:
            return
        self.diag["extra_batches"] += 1
        self.diag["extra_whole_flights"] += len(skipped)
        loss, legal, _, scores, _ = self.trial(skipped, long=True)
        ids = np.flatnonzero(legal)
        note["extra_candidates"] = len(skipped)
        note["extra_landed"] = len(ids)
        if not len(ids):
            return
        self.diag["formerly_skipped_legal"] += len(ids)
        ids = ids[np.argsort(loss[ids])[:5]]
        votes, risk = self.uncertainty([skipped[i] for i in ids], loss[ids])
        possible = np.flatnonzero(votes >= 1)
        if len(possible):
            k = possible[np.argmin(risk[possible])]
            self.tick, self.best, self.lease = tick, skipped[ids[k]], course.mpc.LEASE
            note["extra_pick"] = {
                "votes": int(votes[k]),
                "risk": float(risk[k]),
                "radius": float(scores[ids[k], 12]),
            }
            self.diag["rescans_issued"] += 1


def run(a):
    with patch.object(course.mpc, "MPC", Unscreened):
        course.run(a)
    d = _read(a.out)
    d["screen_sha256"] = _hash(Path(__file__))
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--carry", type=Path, default=course.ROOT / "learning-12.json.gz")
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(mode="frozen", deck="assessment", cap=240, resume=False)
    run(p.parse_args())
