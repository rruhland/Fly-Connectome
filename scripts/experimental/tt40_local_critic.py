"""Learn a bounded local correction to a previously successful motor-outcome model."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt38_motor_value import SCALE, MotorValue, features
from .tt39_skilled_memory import OBS_SCALE


def collect(files, out):
    examples = {}
    used = set()
    for src in files:
        d = _read(src)
        if len(d["rows"]) != len(d["cards"]):
            raise ValueError("Incomplete practice course")
        for c, r in zip(d["cards"], d["rows"], strict=True):
            used.add(c["id"])
            moves = r["records"]
            events = [
                (e["time"], e["side"])
                for e in r["events"]
                if e["kind"] == "legal_return"
            ]
            corrections = [
                s["tick"] for s in r.get("servo", []) if s["status"] == "corrected"
            ]
            for i, m in enumerate(moves):
                if (
                    m["score"] is None
                    or m.get("impact_margin")
                    or m.get("camera_rescue")
                    or (m.get("late_repair") or {}).get("status") == "corrected"
                ):
                    continue
                tick = m["tick"]
                end = moves[i + 1]["tick"] if i + 1 < len(moves) else len(r["actions"])
                if end - tick < 23 or any(
                    tick < t < min(end, tick + 22) for t in corrections
                ):
                    continue
                near = next(
                    (t for t, k in events if k == 1 and tick / 60 < t <= end / 60), None
                )
                answered = near is not None and any(
                    t > near and k == -1 for t, k in events
                )
                if (
                    near is None
                    and i + 1 == len(moves)
                    and r["events"][-1]["kind"] != "point"
                ):
                    continue
                key = (
                    c["id"],
                    c["seed"],
                    c["reply_seed"],
                    tick,
                    *np.round(m["context"], 5),
                    *np.round(m["score"], 5),
                )
                examples[key] = [
                    m["context"],
                    features(m["score"])[0].tolist(),
                    [int(near is not None), int(answered)],
                ]
    _write(
        out,
        {
            "examples": list(examples.values()),
            "ids": sorted(used),
            "source_sha256": {str(p): _hash(p) for p in files},
        },
    )


class Conditional(MotorValue):
    def __init__(self, base, data, update=True):
        super().__init__(base)
        self.examples = list(data["examples"])
        self.update = update
        self.ctx = None
        self.online = []
        self.rebuild()
        self.begin_point()

    def rebuild(self):
        rows = self.examples + self.online[-400:]
        if not rows:
            self.z = np.empty((0, 10))
            return
        state, act, y = map(np.asarray, zip(*rows, strict=True))
        self.z = state / OBS_SCALE
        self.act = act / SCALE
        self.resid = y - self.predict_features(act)
        self.importance = np.ones(len(rows))
        self.importance[len(self.examples) :] = 1.3

    def context(self, x):
        self.ctx = np.array(x, copy=True)

    def predict(self, outcomes):
        f = features(outcomes)
        p = self.predict_features(f)
        if self.ctx is None or not len(self.z):
            return p
        dx = np.sum(((self.ctx / OBS_SCALE) - self.z) ** 2, axis=1)
        ix = np.argsort(dx)[:64]
        ws = np.exp(-0.32 * dx[ix]) * self.importance[ix]
        da = np.mean((f[:, None, :] / SCALE - self.act[ix]) ** 2, axis=2)
        w = ws[None, :] * np.exp(-1.45 * da)
        return np.clip(
            p
            + np.clip(
                w @ self.resid[ix] / (3 + w.sum(axis=1, keepdims=True)), -0.32, 0.32
            ),
            0.01,
            0.99,
        )

    def cost(self, outcomes):
        p = self.predict(outcomes)
        return 7 * (1 - p[:, 0]) + 4 * (1 - p[:, 1]) + 0.23 * outcomes[:, 0]

    def begin_point(self):
        self.pending = None
        self.awaiting = None

    def issued(self, score):
        self.pending = (
            None
            if score is None or self.ctx is None
            else [self.ctx.tolist(), features(score)[0].tolist()]
        )

    def feedback(self, sides, terminal):
        labels = []
        for side in sides:
            if side == 1:
                self.awaiting, self.pending = self.pending, None
            elif self.awaiting is not None:
                labels.append([*self.awaiting, [1, 1]])
                self.awaiting = None
        if terminal:
            if self.awaiting is not None:
                labels.append([*self.awaiting, [1, 0]])
            if self.pending is not None:
                labels.append([*self.pending, [0, 0]])
            self.begin_point()
        if labels and self.update:
            self.online.extend(labels)
            self.rebuild()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("sources", type=Path, nargs="+")
    a = p.parse_args()
    collect(a.sources, a.out)
