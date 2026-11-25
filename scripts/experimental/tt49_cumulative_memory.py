"""Keep all earned joint movements and recover the exact history evicted from smaller checkpoints."""

import argparse
from ast import literal_eval
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt46_continuous_season as season
from . import tt47_season_retention as retention
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt49_memory")


def load_rows(doc):
    p = doc.with_suffix("").with_suffix(".npz")
    meta = _read(doc)
    if _hash(p) != meta["checkpoint_sha256"]:
        raise ValueError("Corrupt motor checkpoint")
    with np.load(p) as z:
        return list(
            zip(z["states"], z["goals"], map(literal_eval, z["tags"]), strict=True)
        )


def identical(a, b):
    return a[2] == b[2] and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def restore_history(end):
    base = load_rows(season.CARRY)
    complete = base.copy()
    proof = []
    previous = 0
    for n in range(12, end + 1, 12):
        doc = season.ROOT / f"learning-{n:02d}.json.gz"
        meta = _read(doc)
        if meta["plan_sha256"] != _hash(season.PLAN) or meta["carry_sha256"] != _hash(
            season.CARRY
        ):
            raise ValueError("Changed practice")
        scored = sum(
            r["planner"].get("credited_fragments", 0) for r in meta["rows"][previous:n]
        )
        rows = load_rows(doc)
        if not 0 <= scored <= len(rows):
            raise ValueError("Snapshot interval lost an entire moving window")
        overlap = len(rows) - scored
        if overlap and not all(
            identical(a, b)
            for a, b in zip(complete[-overlap:], rows[:overlap], strict=True)
        ):
            raise ValueError("Recovered practice intervals do not meet exactly")
        complete.extend(rows[overlap:])
        proof.append(
            dict(through=n, added=scored, overlap=overlap, document=_hash(doc))
        )
        previous = n
    return complete, proof


class Growing(course.Carried):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.online = deque(self.online)
        self.states = None
        self.stamp = None

    def priors(self, x, q, ref, limit):
        key = course.mpc.query(x, q, ref)
        stamp = (len(self.online), id(self.online[-1][0]) if self.online else 0)
        if stamp != self.stamp:
            self.stamp = stamp
            self.arr = list(self.online)
            self.states = np.asarray([r[0] for r in self.arr]).reshape(-1, len(key))
        near, ids = self.tree.query(key, min(len(self.rows), 160))
        combined = [
            (ds, self.rows[j][1], self.rows[j][2], False)
            for ds, j in zip(near, ids, strict=True)
        ]
        d = np.maximum(0.0, np.linalg.norm(self.states - key, axis=1) - 0.10)
        # Within each bank the earliest representative of a tied episode wins,
        # matching the existing nearest-neighbor policy and deterministic ties.
        tagged = set()
        for j in np.argsort(d, kind="stable"):
            _, g, tag = self.arr[j]
            if tag not in tagged:
                combined.append((d[j], g, tag, True))
                tagged.add(tag)
                if len(tagged) == limit:
                    break
        chosen = []
        used = set()
        for distance, g, tag, online in sorted(combined, key=lambda r: r[0]):
            if tag in used:
                continue
            if len(chosen) >= 2 and distance > 3.1:
                break
            chosen.append((g.copy(), online))
            used.add(tag)
            if len(chosen) == limit:
                break
        return chosen


def run(a):
    proof = []
    if a.mode == "frozen":
        whole, proof = restore_history(a.through)

        def restore(model, p, doc):
            retention.ancestral_restore(model, p, doc, season.CARRY)
            model.priors.online = deque(whole)
    else:
        restore = course.resume
    with (
        patch.object(course, "Carried", Growing),
        patch.object(course, "resume", restore),
    ):
        season.run(a)
    doc = _read(a.out)
    doc.update(history=proof, policy=_hash(Path(__file__)))
    _write(a.out, doc)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("live", "frozen", "inspect"))
    p.add_argument("out", type=Path)
    p.add_argument("--through", type=int, choices=(12, 24, 36, 48), default=24)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(cap=240, resume=False)
    a = p.parse_args()
    a.out.parent.mkdir(exist_ok=True, parents=True)
    if a.mode == "inspect":
        rows, proof = restore_history(a.through)
        _write(a.out, dict(complete_rows=len(rows), history=proof))
    else:
        a.deck = "assessment" if a.mode == "frozen" else "practice"
        a.carry = (
            season.ROOT / f"learning-{a.through:02d}.json.gz"
            if a.mode == "frozen"
            else season.CARRY
        )
        run(a)
