"""A planner-free direct policy that aligns observed motion with previously rewarded motor sequences."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt37_joint_deep as deep
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_visual_inertia import LearnedOptics
from .tt39_action_value import SENSOR_SCALE


class Clips:
    def __init__(self):
        self.x = []
        self.u = []
        self.episode = []
        self.tick = []
        self.count = 0
        self.dirty = True

    def append(self, x, u, t):
        if len(t) < 3:
            return
        self.x.extend(x)
        self.u.extend(u)
        self.tick.extend(t)
        self.episode.extend([self.count] * len(t))
        self.count += 1
        self.dirty = True

    def load(self, path):
        doc = _read(path.with_suffix(".json"))
        if _hash(path) != doc["bank_sha256"]:
            raise ValueError("Changed camera motor bank")
        with np.load(path) as z:
            states = z["states"]
            labels = z["targets_now"]
            marks = z["marks"]
            start = 0
            for r in doc["practice"]:
                end = start + r["frames"]
                m = marks[start:end]
                ids = np.flatnonzero(
                    (m[:, 1] < -0.012)
                    & (m[:, 1] > -0.60)
                    & (states[start:end, 4] < -0.9)
                )
                stamp = np.rint((m[ids, 0] / 60 - m[ids, 1]) * 480)
                for hit in np.unique(stamp):
                    sel = ids[stamp == hit]
                    keep = np.r_[True, np.diff(m[sel, 0]) > 0]
                    sel = sel[keep]
                    self.append(
                        states[start + sel], labels[start + sel], m[sel, 0].astype(int)
                    )
                start = end

    def index(self):
        if self.dirty and self.x:
            self.states = np.asarray(self.x)
            self.commands = np.asarray(self.u)
            self.episodes = np.asarray(self.episode)
            self.ticks = np.asarray(self.tick)
            self.tree = cKDTree(self.states / SENSOR_SCALE)
            self.dirty = False

    def candidates(self, center, n):
        if not self.x:
            return []
        self.index()
        d, i = self.tree.query(center, min(n, len(self.x)))
        return [
            (self, j, float(dx))
            for dx, j in zip(np.atleast_1d(d), np.atleast_1d(i), strict=True)
        ]


class SequencePolicy:
    def __init__(self, base, demos, learn):
        self.base = base
        self.demos = demos
        self.earned = Clips()
        self.learn = learn
        self.failures = deque(maxlen=120)
        self.serial = 0
        self.begin_point()

    def begin_point(self):
        self.score = 0
        self.tick = 0
        self.previous_tick = -200
        self.belief = {}
        self.history = []
        self.serial += 1
        self.diagnostics = {
            "sequence_actions": 0,
            "earned_clips": self.earned.count,
            "failed_clips": len(self.failures),
        }

    def clock_tick(self, tick):
        self.tick = tick

    def get(self, optical, q, *, reference=None):
        base, conf = self.base.get(optical, q, reference=reference)
        if optical[4] > -0.9:
            self.belief.clear()
            return base, conf
        center = np.r_[optical, q[:5]] / SENSOR_SCALE
        options = self.demos.candidates(center, 50) + self.earned.candidates(center, 18)
        if not options:
            return base, conf
        step = self.tick - self.previous_tick
        self.previous_tick = self.tick
        if step > 6:
            self.belief.clear()
        distinct = {}
        for lib, i, d in options:
            ident = (lib is self.earned, lib.episodes[i])
            cost = d * d
            if ident in self.belief:
                prev, old = self.belief[ident]
                shift = lib.ticks[i] - prev - step
                cost += (
                    0.19 * min(abs(shift), 8)
                    - 0.72 * np.exp(-0.12 * shift**2)
                    + 0.10 * old
                )
            # Mode continuity favors clips actually following the observed
            # trajectory, while a new demonstration may enter on any frame.
            if ident not in distinct or cost < distinct[ident][0]:
                distinct[ident] = (cost, lib, i, d)
        best = sorted(distinct.items(), key=lambda r: r[1][0])[:7]
        self.belief = {tag: (lib.ticks[i], cost) for tag, (cost, lib, i, _) in best}
        if best[0][1][3] > 1.95:
            return base, conf
        options = []
        for _, (cost, lib, i, _) in best:
            cmd = lib.commands[i].copy()
            cmd[:5] += 0.55 * np.clip(q[:5] - lib.states[i, 6:11], -0.24, 0.24)
            risk = 0.0
            if self.failures:
                # An earlier miss downweights only a nearby sensory/action pair.
                for x, u in self.failures:
                    err = np.mean(((cmd[:5] - q[:5] - u) / 0.14) ** 2)
                    risk = max(
                        risk, np.exp(-0.7 * np.sum((center - x) ** 2) - 0.6 * err)
                    )
            options.append((cost + 1.45 * risk, cmd))
        # Average closely aligned movements in a single local motor mode.
        k = int(np.argmin([v for v, _ in options]))
        anchor = options[k][1]
        weights = np.array(
            [
                np.exp(
                    -1.8 * (v - options[k][0])
                    - 2 * np.mean(((cmd[:5] - anchor[:5]) / 0.19) ** 2)
                )
                for v, cmd in options
            ]
        )
        chosen = np.average([u for _, u in options], axis=0, weights=weights)
        self.diagnostics["sequence_actions"] += 1
        return (chosen if base is None else 0.45 * base + 0.55 * chosen), max(
            conf, 0.93
        )

    def record_command(self, tick, x, q, ref):
        if x is None or x[4] >= -0.9:
            return
        if self.history and tick - self.history[-1][0] > 18:
            self.history = []
        self.history.append((tick, np.r_[x, q[:5]], ref.copy()))

    def finish(self, success):
        if self.learn and len(self.history) >= 6:
            rows = self.history[-40:]
            if success:
                self.earned.append(
                    [x for _, x, _ in rows],
                    [r for _, _, r in rows],
                    [t for t, _, _ in rows],
                )
            else:
                self.failures.extend(
                    (x / SENSOR_SCALE, r[:5] - x[6:11]) for _, x, r in rows[::4]
                )
        self.belief = {}
        self.history = []

    def score_feedback(self, score):
        if score > self.score:
            self.finish(True)
            self.score = score

    def end_point(self, terminal):
        if terminal:
            self.finish(False)
        self.diagnostics.update(
            earned_clips=self.earned.count, failed_observations=len(self.failures)
        )


def run(a):
    used = {
        r["card"]["id"]
        for f in a.demos
        for r in _read(f.with_suffix(".json"))["practice"]
    }
    held = {c["id"] for c in _read(a.plan)[a.deck][: a.limit]}
    if used & held:
        raise ValueError("A demonstration is also in the assessment")
    d = Clips()
    for p in a.demos:
        d.load(p)
    models = []

    def wrap(base):
        actor = SequencePolicy(base, d, a.live)
        models.append(actor)
        return actor

    original = deep.Vision

    class Eye(original):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.eye = LearnedOptics(self.eye, a.optics)

    with patch.object(deep, "Vision", Eye):
        deep.run(
            a.feeds,
            a.bank,
            a.plan,
            a.deck,
            "local",
            a.out,
            cap=240,
            limit=a.limit,
            field_hook=wrap,
            details={
                "motor_demos": {str(p): _hash(p) for p in a.demos},
                "online": a.live,
                "demonstrated_clips": d.count,
            },
        )
    m = models[0]
    ck = a.out.with_suffix("").with_suffix(".npz")
    np.savez_compressed(
        ck,
        states=m.earned.x,
        commands=m.earned.u,
        clip=m.earned.episode,
        tick=m.earned.tick,
        failed_x=[r[0] for r in m.failures],
        failed_action=[r[1] for r in m.failures],
    )
    data = _read(a.out)
    data["checkpoint_sha256"] = _hash(ck)
    _write(a.out, data)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--live", action="store_true")
    p.add_argument(
        "--bank", type=Path, default=Path("artifacts/tt37/self-combined.npz")
    )
    p.add_argument(
        "--demos",
        nargs="+",
        type=Path,
        default=[
            Path("artifacts/tt37/self-combined.npz"),
            Path("artifacts/tt39/earned-01.npz"),
        ],
    )
    p.add_argument("--optics", type=Path, default=Path("artifacts/tt37/rebound.npz"))
    p.add_argument(
        "--plan", type=Path, default=Path("artifacts/tt37/memory-course.json")
    )
    p.add_argument("--deck", default="replication")
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    run(p.parse_args())
