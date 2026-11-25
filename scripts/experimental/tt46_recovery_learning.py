"""Start preparing between strokes from the future joint configurations that earned subsequent scores."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt43_temporal_servo as servo
from . import tt44_rally_course as fresh
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt46_recovery")
BANK = ROOT / "next-position.npz"
mpc = course.mpc
SCALE = np.array([0.19, 0.29, 0.13, 0.70, 1.25, 1.0, 0.6, 0.55, 0.55, 0.60, 0.65])


def passage(x):
    return x is not None and x[4] > 0.9 and -0.65 < x[1] < 0.15


def prep(x):
    return x is not None and x[4] < -0.9 and -0.62 < x[1] < -0.34


def fit(out):
    keys, targets, groups = [], [], []
    for path in servo.SOURCES:
        receipt = _read(path.with_suffix(".json"))
        if receipt["bank_sha256"] != _hash(path):
            raise ValueError("Practice state checksum mismatch")
        with np.load(path) as z:
            states, marks = z["states"], z["marks"]
        start = 0
        for row in receipt["practice"]:
            end = start + row["frames"]
            episode = states[start:end]
            meta = marks[start:end]
            start = end
            # Each window belongs to one earlier scored physical hit. The next
            # known hit must be the immediately following actual successful hit.
            stamps = np.rint((meta[:, 0] * mpc.DT - meta[:, 1]) * 480).astype(int)
            groups_in_time = np.unique(stamps)
            for prev, new in zip(groups_in_time, groups_in_time[1:]):
                if not 420 < new - prev < 1250:
                    continue
                before = np.flatnonzero(
                    (stamps == prev)
                    & (meta[:, 1] > 0.0)
                    & (episode[:, 4] > 0.9)
                    & (episode[:, 1] > -0.65)
                    & (episode[:, 1] < 0.15)
                )
                after = np.flatnonzero(
                    (stamps == new)
                    & (meta[:, 1] < -0.04)
                    & (episode[:, 4] < -0.9)
                    & (episode[:, 1] > -0.62)
                    & (episode[:, 1] < -0.34)
                )
                if len(before) and len(after):
                    keys.append(episode[before[0]])
                    targets.append(episode[after[0], 6:11])
                    groups.append(row["card"]["id"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, keys=keys, targets=targets, groups=groups)
    _write(
        out.with_suffix(".json"),
        dict(
            bank=_hash(out),
            inputs={str(p): _hash(p) for p in servo.SOURCES},
            count=len(keys),
            releases=sorted(set(groups)),
        ),
    )


class Preparation:
    def __init__(self, keys, targets):
        self.pairs = deque(zip(keys, targets, strict=True), maxlen=700)

    def next(self, key, home):
        if not self.pairs:
            return home.copy(), 0
        keys, targets = map(np.asarray, zip(*self.pairs))
        distance = np.mean(np.minimum(((keys - key) / SCALE) ** 2, 16.0), axis=1)
        ids = np.argsort(distance)[:16]
        weights = np.exp(-2.5 * distance[ids])
        if weights.sum() < 2.5:
            return home.copy(), 0
        # Choose an observed point close to the locally likely future states.
        # Penalize long moves when the alternatives cover the same support.
        tgt = targets[ids]
        cross = np.sqrt(np.mean((tgt[:, None] - tgt[None, :]) ** 2, axis=2))
        expense = cross @ weights / weights.sum() + 0.08 * np.linalg.norm(
            tgt - home, axis=1
        )
        return tgt[np.argmin(expense)], len(ids)


class Actor(servo.Temporal):
    def __init__(self, *args, fit, **kw):
        super().__init__(*args, **kw)
        self.fit = fit
        self.priors.confirm = self.confirm

    def begin(self):
        super().begin()
        self.last = None
        self.pending = None
        self.seen_in = False
        self.opened = False
        self.sampled = False
        self.ready = None
        self.notes_recovery = []

    def step(self, tick, eye, x, q, reference, base):
        desired = super().step(tick, eye, x, q, reference, base)
        if x is None:
            return desired
        if x[4] < -0.9:
            self.ready = None
            self.opened = False
            self.seen_in = True
            if self.last is not None and not self.sampled and prep(x):
                self.pending = (self.last.copy(), q[:5].copy())
                self.sampled = True
            return desired
        if self.seen_in and not self.opened and passage(x):
            self.last = np.r_[x, q[:5]]
            self.ready, n = self.fit.next(self.last, self.priors.home[:5])
            self.opened = True
            self.sampled = False
            self.seen_in = False
            self.notes_recovery.append(
                dict(tick=tick, neighbors=n, goal=self.ready.tolist())
            )
        if self.ready is not None and x[4] > 0.9:
            desired = desired.copy()
            desired[:5] += 0.45 * np.clip(
                self.ready - self.priors.home[:5], -0.50, 0.50
            )
        return desired

    def confirm(self):
        if self.pending is not None:
            self.fit.pairs.append(self.pending)
            self.pending = None
            self.diag["earned_next_position"] += 1


class Credit(course.Carried):
    def keep(self, archive, leg):
        self.confirm()
        return super().keep(archive, leg)


def run(a):
    receipt = _read(BANK.with_suffix(".json"))
    if receipt["bank"] != _hash(BANK) or set(receipt["releases"]) & {
        c["id"] for c in _read(fresh.PLAN)[a.deck][: a.n]
    }:
        raise ValueError("Release overlap or corrupted memory")
    with np.load(BANK) as f:
        learned = Preparation(f["keys"], f["targets"])
    native = mpc.play

    def play(*args, **kw):
        row = native(*args, **kw)
        row["optical_recovery"] = args[10].notes_recovery
        return row

    with (
        patch.object(course, "PLAN", fresh.PLAN),
        patch.object(course, "Carried", Credit),
        patch.object(mpc, "MPC", lambda *args, **kw: Actor(*args, fit=learned, **kw)),
        patch.object(mpc, "play", play),
    ):
        course.run(a)
    record = _read(a.out)
    record.update(
        preparation=_hash(BANK),
        learned=[(k.tolist(), q.tolist()) for k, q in learned.pairs],
    )
    _write(a.out, record)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "play"))
    p.add_argument("out", type=Path)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--n", type=int, default=12)
    p.set_defaults(
        deck="assessment", mode="live", cap=240, resume=False, carry=fresh.CARRY
    )
    a = p.parse_args()
    fit(a.out) if a.phase == "fit" else run(a)
