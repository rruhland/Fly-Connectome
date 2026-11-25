"""Learn a pre-reception joint posture that leaves time for several plausible rewarded movements."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt43_temporal_servo as servo
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from . import tt46_recovery_learning as old
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt49_body")
MODEL = ROOT / "transitions.npz"
DT = course.mpc.DT


def fit():
    rows = []
    for p in servo.SOURCES:
        d = _read(p.with_suffix(".json"))
        if d["bank_sha256"] != _hash(p):
            raise ValueError("Training camera bank changed")
        with np.load(p) as z:
            states, marks = z["states"], z["marks"]
        start = 0
        for row in d["practice"]:
            end = start + row["frames"]
            e, m = states[start:end], marks[start:end]
            start = end
            stamps = np.rint((m[:, 0] * DT - m[:, 1]) * 480).astype(int)
            times = np.unique(stamps)
            for a, b in zip(times, times[1:]):
                if not 420 < b - a < 1250:
                    continue
                release = np.flatnonzero(
                    (stamps == a)
                    & (e[:, 4] > 0.9)
                    & (e[:, 1] > -0.65)
                    & (e[:, 1] < 0.15)
                )
                approach = np.flatnonzero(
                    (stamps == b)
                    & (e[:, 4] < -0.9)
                    & (e[:, 1] > -0.62)
                    & (e[:, 1] < -0.34)
                )
                late = np.flatnonzero(
                    (stamps == b)
                    & (e[:, 4] < -0.9)
                    & (m[:, 1] < -0.035)
                    & (m[:, 1] > -0.095)
                )
                if not len(release) or not len(approach) or not len(late):
                    continue
                i, j, k = release[0], approach[0], late[-1]
                deadline = (m[k, 0] - m[j, 0]) * DT
                if deadline > 0.06:
                    rows.append(
                        (e[i], e[j, 6:11], e[k, 6:11], deadline, row["card"]["id"])
                    )
    keys, pre, late, times, ids = map(np.asarray, zip(*rows))
    rates = np.maximum(
        0.18, np.quantile(np.abs(late - pre) / times[:, None], 0.85, axis=0)
    )
    ROOT.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(MODEL, keys=keys, pre=pre, late=late, times=times, rates=rates)
    _write(
        MODEL.with_suffix(".json"),
        dict(
            model=_hash(MODEL),
            sources={str(p): _hash(p) for p in servo.SOURCES},
            release_ids=np.unique(ids).tolist(),
            samples=len(keys),
            learned_joint_rates=rates.tolist(),
        ),
    )


class Readiness:
    def __init__(self):
        with np.load(MODEL) as z:
            self.exp = deque(
                zip(z["keys"], z["pre"], z["late"], z["times"], strict=True), maxlen=900
            )
            self.rates = z["rates"]

    def select(self, key, home):
        keys, pre, late, times = map(np.asarray, zip(*self.exp))
        d = np.mean(np.minimum(((keys - key) / old.SCALE) ** 2, 16), axis=1)
        ix = np.argsort(d)[:26]
        w = np.exp(-2.6 * d[ix])
        if w.sum() < 3.0:
            return home.copy(), 0.0
        goals = late[ix]
        # Draw preparation poses from real prior passages and keep home as a
        # feasible default. Joint speed limits and the observed time to reach
        # the later scored configuration determine the tolerated displacement.
        options = np.vstack([home, pre[ix], home + 0.55 * (pre[ix] - home)])
        travel = np.max(np.abs(options[:, None] - goals) / self.rates, axis=2)
        margin = travel - times[ix]
        excess = np.maximum(margin + 0.025, 0.0)
        mean = excess @ w / w.sum()
        # Tail cost conditional on the observed set of possible replies.
        order = np.argsort(-excess, axis=1)
        rank_w = w[order]
        before = np.cumsum(rank_w, axis=1) - rank_w
        mass = np.clip(0.25 * w.sum() - before, 0.0, rank_w)
        worst = np.sum(mass * np.take_along_axis(excess, order, axis=1), axis=1) / (
            0.25 * w.sum()
        )
        effort = np.sqrt(np.mean(((options - home) / self.rates) ** 2, axis=1))
        loss = 0.4 * mean + 0.6 * worst + 0.15 * effort
        j = np.argmin(loss)
        return options[j].copy(), float(loss[0] - loss[j])


class Memory(course.Carried):
    def keep(self, tape, leg):
        self.confirm(tape)
        return super().keep(tape, leg)


class Player(season.Player):
    def __init__(self, *a, readiness, **kw):
        super().__init__(*a, **kw)
        self.readiness = readiness
        self.priors.confirm = self.confirm

    def begin(self):
        super().begin()
        self.sent = None
        self.last = None
        self.first = None
        self.seen_in = False
        self.flying = False
        self.pending_reward = None
        self.gain = 0.0
        self.recovery_log = []

    def step(self, tick, eye, x, q, ref, base):
        u = super().step(tick, eye, x, q, ref, base)
        if x is not None and x[4] < -0.9:
            self.seen_in = True
            self.flying = False
            if self.last is not None and self.first is None and old.prep(x):
                self.first = (tick, q[:5].copy())
        if self.seen_in and old.passage(x):
            self.seen_in = False
            self.flying = True
            if self.last is not None and self.first is not None:
                self.pending_reward = (self.last, self.first, tick)
            self.last = np.r_[x, q[:5]]
            self.first = None
            self.sent, self.gain = self.readiness.select(
                self.last, self.priors.home[:5]
            )
            self.recovery_log.append([tick, self.gain, self.sent.tolist()])
        if self.flying and self.gain > 0.012 and (x is None or x[4] > 0.9):
            # The normal sensor-controlled arm can move towards the selected
            # readiness as soon as the outgoing stroke is visible.
            weight = min(0.75, 0.22 + 3.0 * self.gain)
            u = u.copy()
            u[:5] += weight * np.clip(self.sent - self.priors.home[:5], -0.48, 0.48)
        return u

    def confirm(self, tape):
        if self.pending_reward is not None:
            source, (first, early), end = self.pending_reward
            self.pending_reward = None
        elif self.last is not None and self.first is not None:
            source, (first, early), end = self.last, self.first, tape[-1][0] + 1
            self.first = None
        else:
            return
        recent = [
            (t, q)
            for t, x, q, *_ in tape
            if first < t < end and x is not None and x[4] < -0.9 and x[1] < -0.87
        ]
        if len(recent) < 4:
            return
        t, late = recent[-3]
        dt = (t - first) * DT
        if 0.06 < dt < 0.65:
            self.readiness.exp.append((source.copy(), early, late[:5].copy(), dt))
            self.diag["earned_readiness_transitions"] += 1


def run(a):
    model = _read(MODEL.with_suffix(".json"))
    if model["model"] != _hash(MODEL) or set(model["release_ids"]) & {
        c["id"] for c in _read(season.PLAN)[a.deck]
    }:
        raise ValueError("Changed model or overlapping release")
    fitment = Readiness()
    native = course.mpc.play

    def play(*args, **kw):
        row = native(*args, **kw)
        row["readiness"] = args[10].recovery_log
        return row

    with (
        patch.object(course, "Carried", Memory),
        patch.object(course.mpc, "play", play),
        patch.object(
            season, "Player", lambda *args, **kw: Player(*args, readiness=fitment, **kw)
        ),
    ):
        season.run(a)
    doc = _read(a.out)
    doc.update(
        model=_hash(MODEL),
        examples=[
            (k.tolist(), p.tolist(), q.tolist(), t) for k, p, q, t in fitment.exp
        ],
    )
    _write(a.out, doc)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("fit", "live"))
    p.add_argument("out", type=Path, nargs="?")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        deck="practice", mode="live", carry=season.CARRY, cap=240, resume=False
    )
    a = p.parse_args()
    fit() if a.action == "fit" else run(a)
