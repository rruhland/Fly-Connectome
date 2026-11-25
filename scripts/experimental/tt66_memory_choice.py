"""Learn which retained joint memory to consult on each visually observed incoming ball."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
from . import tt49_cumulative_memory as storage
from . import tt52_practice_dose as dose
from .tt4_transfer_trial import _hash, _read, _write
from .tt54_training_speedups import enabled

ROOT = Path("artifacts/tt66_memory_choice")
PLAN = ROOT / "protocol.json"
course = dose.course
season = dose.season
MP = course.mpc
SOURCES = [course.ROOT / f"learning-{n:02}.json.gz" for n in (12, 36, 48)]


def plan():
    ids = {c["id"] for c in _read(dose.PLAN)["practice"]}
    for p in SOURCES:
        d = _read(p)
        if ids & {c["id"] for c in d["cards"][: len(d["rows"])]}:
            raise ValueError("Release overlap")
    ROOT.mkdir(parents=True, exist_ok=True)
    _write(
        PLAN,
        dict(
            schedule=_hash(dose.PLAN),
            parents={str(p): _hash(p) for p in SOURCES},
            n=96,
            learning_comparison_n=64,
            rule="Three historical timescales of the same motor policy plus its changing recent memory. One expert per incoming ball. The main memory updates on every paid return. Fixed comparison samples uniformly from the same options; the adaptive option learns from real own return and the next actual return. All 96 points per adaptive learner use independently sampled continuous opponent requests; first 64 also paired to uniform selection.",
        ),
    )


def key(x, q):
    return np.r_[
        x[:3] / [0.16, 0.19, 0.07],
        x[3:6] / [0.68, 0.89, 0.82],
        q[:5] / [0.7, 0.6, 0.7, 0.75, 0.8],
    ]


class Selector:
    def __init__(self, kind):
        self.kind = kind
        self.rng = np.random.default_rng(660523)
        self.rows = deque(maxlen=512)
        self.history = []
        self.begin()

    def begin(self):
        self.active = None
        self.previous = None

    def select(self, s):
        if self.active is not None:
            return self.active[0]
        prior = np.tile([1.1, 5.2], (4, 1))
        if self.rows:
            states, choices, won, credit = map(np.asarray, zip(*self.rows, strict=True))
            dist = np.sum((states - s) ** 2, axis=1)
            w = credit * (0.09 + 0.91 * np.exp(-np.minimum(15, dist) * 0.62))
            np.add.at(prior, (choices, won), w)
        if self.kind == "adaptive":
            option = int(np.argmax(self.rng.beta(prior[:, 1], prior[:, 0])))
        else:
            option = int(self.rng.integers(4))
        self.active = (option, s.copy())
        return option

    def record(self, won):
        for v, credit in [(self.active, 0.65), (self.previous, 0.35)]:
            if v is not None:
                option, s = v
                self.rows.append((s, option, int(won), credit))
        if self.active is not None:
            self.history.append([self.active[0], int(won)])
        self.previous = self.active if won else None
        self.active = None


class Player(season.Player):
    def __init__(self, *args, selector, **kw):
        super().__init__(*args, **kw)
        self.selector = selector
        self.books = [storage.load_rows(p) for p in SOURCES]
        ordinary = self.priors.priors

        def lookup(x, q, ref, limit):
            i = self.selector.select(key(x, q))
            if i == 0:
                return ordinary(x, q, ref, limit)
            # Keep the last newly earned movements available at every timescale.
            live = self.priors.online
            initial = self.books[i - 1]
            recent = [
                entry
                for entry in list(live)[-48:]
                if entry[2][0] == "live" and entry[2][1] > 48
            ]
            self.priors.online = initial + recent
            try:
                return ordinary(x, q, ref, limit)
            finally:
                self.priors.online = live

        self.priors.priors = lookup
        old = self.priors.keep

        def paid(tape, tag):
            self.selector.record(1)
            return old(tape, tag)

        self.priors.keep = paid

    def begin(self):
        super().begin()
        self.selector.begin()

    def missed(self, tape):
        self.selector.record(0)


def run(a):
    p = _read(PLAN)
    if p["schedule"] != _hash(dose.PLAN) or any(
        _hash(Path(k)) != v for k, v in p["parents"].items()
    ):
        raise ValueError("Altered experience")
    rank = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    selector = Selector(a.kind)
    original = MP.play

    def game(*args, **kw):
        previous = len(selector.history)
        r = original(*args, **kw)
        r["memory_choices"] = selector.history[previous:]
        return r

    a.n = p["n" if a.kind == "adaptive" else "learning_comparison_n"]
    a.out = ROOT / (a.kind + ".json.gz")
    with (
        enabled(),
        patch.object(course, "PLAN", dose.PLAN),
        patch.object(MP, "play", game),
        patch.object(
            MP,
            "MPC",
            lambda *args, **kw: Player(*args, selector=selector, coverage=rank, **kw),
        ),
    ):
        course.run(a)
    _write(
        ROOT / (a.kind + "-selector.json"),
        dict(
            rows=[[s.tolist(), i, w, c] for s, i, w, c in selector.rows],
            rng=selector.rng.bit_generator.state,
            checkpoint=_hash(a.out.with_suffix("").with_suffix(".npz")),
            protocol=_hash(PLAN),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("plan", "adaptive", "uniform"))
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        mode="live", deck="practice", resume=False, cap=240, carry=season.CARRY
    )
    a = p.parse_args()
    plan() if a.kind == "plan" else run(a)
