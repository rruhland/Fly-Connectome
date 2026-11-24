"""Learn direct motor commands conditioned on observed motion and servo load from real returns."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt37_joint_deep as direct
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_visual_inertia import LearnedOptics
from .tt39_turn_time import turn_time

BASE = Path("artifacts/tt37/self-combined.npz")
SCALE = np.r_[
    direct.SCALES["local"],
    [0.036, 0.030, 0.039, 0.047, 0.045],
    [0.18, 0.12, 0.15, 0.21, 0.18],
]


def demonstrations(source):
    receipt = _read(source.with_suffix(".json"))
    if _hash(source) != receipt["bank_sha256"]:
        raise ValueError("Demonstration hash mismatch")
    with np.load(source) as a:
        x, ref, marks = a["states"], a["targets_now"], a["marks"]
        groups = np.repeat(
            np.arange(len(receipt["practice"])),
            [v["frames"] for v in receipt["practice"]],
        )
        use = (
            np.flatnonzero(
                (groups[:-1] == groups[1:])
                & (np.diff(marks[:, 0]) == 1)
                & (x[1:, 4] < -0.9)
                & (x[:-1, 4] < -0.9)
            )
            + 1
        )
        q = x[use, 6:11]
        states = np.c_[x[use], q - x[use - 1, 6:11], ref[use - 1, :5] - q]
        delta = ref[use, :5] - ref[use - 1, :5]
    return states, delta, {v["card"]["id"] for v in receipt["practice"]}


class Motor:
    def __init__(self, base, examples, mode, learn):
        self.base = base
        self.with_motion = mode == "motion"
        self.learn = learn
        self.n = 21 if self.with_motion else 11
        self.x, self.u = examples
        self.tree = cKDTree(self.encode(self.x))
        self.online = []
        self.begin_point()

    def encode(self, x):
        return x[..., : self.n] / SCALE[: self.n]

    def begin_point(self):
        self.previous = None
        self.tick = -1
        self.pending = []
        self.frame = None
        self.last_score = 0
        self.diag = dict(issued=0, scored=0, unrewarded=0)

    def clock_tick(self, t):
        self.tick = t
        self.frame = None

    def get(self, x, q, *, reference):
        old = self.previous
        self.previous = (self.tick, q[:5].copy())
        usual, confidence = self.base.get(x, q, reference=reference)
        if x[4] >= -0.9 or old is None or old[0] != self.tick - 1:
            return usual, confidence
        state = np.r_[x, q[:5], q[:5] - old[1], reference[:5] - q[:5]]
        key = self.encode(state)
        d, i = self.tree.query(key, k=32)
        labels = self.u[i]
        if self.online:
            od, oi = self.recent.query(key, k=min(len(self.online), 14))
            d = np.r_[d, np.atleast_1d(od)]
            labels = np.vstack([labels, [self.online[j][1] for j in np.atleast_1d(oi)]])
        self.frame = (state, reference[:5].copy())
        rank = np.argsort(d)[:18]
        d, u = d[rank], labels[rank]
        if d[1] > (2.5 if self.with_motion else 1.75):
            return usual, confidence
        w = np.exp(-0.5 * ((d - d[0]) / max(0.20, d[5] - d[0])) ** 2)
        # A small number of measured motor modes can share a similar approach.
        # Retain a coherent mode of issued velocity before averaging its samples.
        similarity = np.exp(
            -0.5 * np.sum(((u[:, None] - u[None, :]) / 0.064) ** 2, axis=2)
        )
        a = np.argmax(similarity @ w)
        w *= similarity[a]
        du = w @ u / w.sum()
        # Absolute setpoints give a restoring term when the body drifts far from
        # the practiced manifold. Bound its authority in dense observed regions.
        inc = (
            du
            if usual is None
            else 0.79 * du + 0.21 * np.clip(usual[:5] - reference[:5], -0.069, 0.069)
        )
        self.diag["issued"] += 1
        return np.r_[reference[:5] + inc, -0.1], 1.0

    def record_command(self, tick, optical, q, ref):
        if self.frame is not None:
            state, prev = self.frame
            self.pending.append((tick, state, ref[:5].copy() - prev))
        elif optical is not None:
            self.pending.append((tick, np.r_[optical, q[:5], np.zeros(10)], None))
        self.pending = self.pending[-125:]

    def score_feedback(self, n):
        if n <= self.last_score:
            return
        self.last_score = n
        self.diag["scored"] += 1
        if self.learn:
            hit = turn_time([(t, s[:6]) for t, s, _ in self.pending])
            if hit is not None:
                self.online.extend(
                    (x, u)
                    for t, x, u in self.pending
                    if u is not None and hit - 29 <= t <= hit - 1
                )
                self.online = self.online[-5500:]
                if self.online:
                    self.recent = cKDTree(
                        self.encode(np.asarray([x for x, _ in self.online]))
                    )
        self.pending = []

    def end_point(self, *, terminal):
        self.diag["unrewarded"] = int(terminal and len(self.pending) > 0)
        self.diagnostics = dict(self.diag, stored=len(self.online))


def run(a):
    cases = _read(a.plan)[a.deck][: a.limit]
    chunks = [demonstrations(z) for z in a.banks]
    used = set.union(*(c[2] for c in chunks))
    if used & {c["id"] for c in cases}:
        raise ValueError("A test release occurs among the demonstrations")
    data = tuple(np.concatenate([c[j] for c in chunks]) for j in range(2))
    instances = []

    def wrap(base):
        obj = Motor(base, data, a.mode, a.learn)
        instances.append(obj)
        return obj

    original = direct.Vision

    class Eye(original):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.eye = LearnedOptics(self.eye, a.optics)

    with patch.object(direct, "Vision", Eye):
        direct.run(
            a.feeds,
            BASE,
            a.plan,
            a.deck,
            "local",
            a.out,
            cap=200,
            limit=a.limit,
            field_hook=wrap,
            details=dict(
                mode=a.mode,
                learn=a.learn,
                practice=sorted(used),
                banks={z.name: _hash(z) for z in a.banks},
            ),
        )
    if a.learn:
        rows = instances[0].online
        ck = a.out.with_suffix("").with_suffix(".npz")
        np.savez_compressed(
            ck,
            states=np.array([x for x, _ in rows]).reshape(-1, 21),
            delta=np.array([u for _, u in rows]).reshape(-1, 5),
        )
        report = _read(a.out)
        report["online_sha256"] = _hash(ck)
        _write(a.out, report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("position", "motion"))
    p.add_argument("out", type=Path)
    p.add_argument("--learn", action="store_true")
    p.add_argument(
        "--banks",
        nargs="+",
        type=Path,
        default=[BASE, Path("artifacts/tt37/camera-mix-rollin.npz")],
    )
    p.add_argument(
        "--plan", type=Path, default=Path("artifacts/tt37/memory-course.json")
    )
    p.add_argument("--deck", default="replication")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--optics", type=Path, default=Path("artifacts/tt37/rebound.npz"))
    run(p.parse_args())
