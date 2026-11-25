"""Learn the downstream value of an outgoing movement using only observed score timing and predicted movement effects."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt43_outgoing_coverage as ranked
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt44_score")
BANK = ROOT / "score-practice.json"


def credited(notes, tick):
    # A landing is delayed. The most recent committed prediction must precede
    # the public score and forecast a departure within 1.25 s of that score.
    for s in reversed(notes):
        if s["tick"] > tick or not s.get("target_errors"):
            continue
        elapsed = (tick - s["tick"]) * course.mpc.DT - s["contact_delay_s"]
        return s["target_errors"] if 0.18 < elapsed < 1.25 else None
    return None


def labels(row):
    # Only public landing side, its clock, and end-of-game are supplied.
    prev = None
    offered = False
    for e in row["events"]:
        if e["kind"] == "legal_return":
            if e["side"] == -1:
                offered = True
            else:
                if prev is not None and offered:
                    yield prev, 0.0
                prev = credited(row["searches"], e["step"] - 1)
                offered = False
        if e["kind"] == "point" and offered and prev is not None:
            yield prev, 1.0
            prev = None


def fit(path):
    validation = {c["id"] for c in _read(course.PLAN)["assessment"]}
    rows = {}
    sources = {}
    for n in [*ranked.TRAIN, "tt42/learning-48"]:
        p = Path(f"artifacts/{n}.json.gz")
        d = _read(p)
        sources[n] = _hash(p)
        for c, r in zip(d["cards"], d["rows"]):
            if c["id"] in validation:
                continue
            for x, y in labels(r):
                key = (c["id"], *np.round(x, 8), y)
                rows[key] = (c["id"], x, y)
    rows = list(rows.values())
    group, x, y = map(np.array, zip(*rows, strict=True))
    if len(np.unique(y)) != 2:
        raise ValueError("Training must include verifiable successes and uncensored failures")
    prediction = np.zeros(len(y))
    for g in np.unique(group):
        test = group == g
        prediction[test] = ranked.estimate(ranked.ridge(x[~test], y[~test]), x[test])
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(
        path,
        {
            "rows": rows,
            "sources": sources,
            "excluded": sorted(validation),
            "validation": {
                "misses": int(y.sum()),
                "releases": len(np.unique(group)),
                "trials": len(y),
                "highest_fifth_misses": int(
                    y[np.argsort(prediction)[-max(1, len(y) // 5) :]].sum()
                ),
                "brier": float(np.mean((y - prediction) ** 2)),
                "constant_brier": float(np.var(y)),
            },
        },
    )


class ScoreModel:
    def __init__(self, packet):
        _, x, y = zip(*packet["rows"], strict=True)
        self.x = list(x)
        self.y = list(y)
        self.count = len(y)
        self.rebuild()

    def rebuild(self):
        x = np.asarray(self.x)
        self.fit = ranked.ridge(x, np.array(self.y))
        self.tree = ranked.cKDTree(x / ranked.SCALE)

    def add(self, x, y):
        if x is None:
            return
        self.x.append(list(x))
        self.y.append(y)
        self.rebuild()

    def risk(self, x):
        d, _ = self.tree.query(x / ranked.SCALE, k=6)
        return ranked.estimate(self.fit, x), d[:, -1] < 1.85


class Actor(ranked.Actor):
    def begin(self):
        super().begin()
        self.prev = None

    def rewarded(self, archive):
        self.coverage.add(self.prev, 0)
        self.prev = credited(self.notes, archive[-1][0])

    def missed(self, archive):
        self.coverage.add(self.prev, 1)
        self.prev = None


def run(args):
    packet = _read(BANK)
    if not {c["id"] for c in _read(course.PLAN)[args.deck][: args.n]} <= set(
        packet["excluded"]
    ):
        raise ValueError("Evaluation release in practice")
    model = ScoreModel(packet)
    original = course.mpc.play

    def play(*pos, **kw):
        actor = pos[10]
        keep = actor.priors.keep

        def reward(history, leg):
            actor.rewarded(history)
            return keep(history, leg)

        with patch.object(actor.priors, "keep", reward):
            r = original(*pos, **kw)
        r["delayed_score_updates"] = len(model.y) - model.count
        r["downstream_choices"] = actor.reranks
        return r

    with (
        patch.object(
            course.mpc, "MPC", lambda *a, **kw: Actor(*a, **kw, coverage=model)
        ),
        patch.object(course.mpc, "play", play),
    ):
        course.run(args)
    d = _read(args.out)
    d.update(
        score_training_sha256=_hash(BANK),
        earned_features=model.x[model.count :],
        earned_misses=model.y[model.count :],
    )
    _write(args.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "play"))
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--mode", choices=("frozen", "live"), default="live")
    p.add_argument("--carry", type=Path)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(deck="assessment", cap=240, resume=False)
    a = p.parse_args()
    fit(a.out) if a.phase == "fit" else run(a)
