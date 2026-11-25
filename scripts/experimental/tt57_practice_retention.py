"""Test what the first 96 continuing games left in the younger motor learner."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt52_practice_dose as dose
from .tt4_transfer_trial import _hash, _read, _write
from .tt54_training_speedups import enabled

course = dose.course
ROOT = Path("artifacts/tt57_practice_retention")
PLAN = ROOT / "fresh.json"
BEFORE = course.ROOT / "learning-12.json.gz"
AFTER = dose.ROOT / "live-from-12.json.gz"


def plan():
    after = _read(AFTER)
    if len(after["rows"]) != 96 or after["carry_sha256"] != _hash(BEFORE):
        raise ValueError("Incomplete 96-game course or changed parent")
    ids = {c["id"] for c in after["cards"]}
    ids.update(c["id"] for c in _read(BEFORE)["cards"][:12])
    candidates = {c["id"]: c for c in _read(dose.season.PLAN)["practice"]}
    if set(candidates) & ids or len(candidates) != 16:
        raise ValueError("The proposed test openings have been used by this learner")
    rng = np.random.default_rng(57129642)
    roster = [
        dict(
            candidates[i],
            seed=int(rng.integers(1, 2**30)),
            reply_seed=int(rng.integers(1, 2**30)),
        )
        for i in rng.permutation(sorted(candidates))
    ]
    ROOT.mkdir(exist_ok=True, parents=True)
    _write(
        PLAN,
        dict(
            assessment=roster,
            practice=roster,
            before=_hash(BEFORE),
            after=_hash(AFTER),
            rule="All 16 independent opening recordings from the older TT46 season, one new opening seed and new stream each, fixed before scoring. These recordings and seeds did not train either checkpoint in this pair. Ordinary broad opponent throughout.",
        ),
    )


def evaluate(a):
    p = _read(PLAN)
    a.carry = BEFORE if a.condition == "before" else AFTER
    if _hash(a.carry) != p[a.condition]:
        raise ValueError("Training boundary changed")
    scorer = dose.season.fresh.shot.Coverage(_read(dose.season.fresh.RISK))
    ordinary = course.resume

    def resume(actor, path, doc):
        n = 12 if a.condition == "after" else 0
        ordinary(actor, path, dict(doc, rows=[None] * n + doc["rows"]))

    a.out = ROOT / (a.condition + ".json.gz")
    with (
        enabled(),
        patch.object(course, "PLAN", PLAN),
        patch.object(course, "resume", resume),
        patch.object(
            course.mpc,
            "MPC",
            lambda *args, **kw: dose.season.Player(*args, coverage=scorer, **kw),
        ),
    ):
        course.run(a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("condition", choices=("plan", "before", "after"))
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(mode="frozen", deck="assessment", n=16, resume=False, cap=240)
    a = p.parse_args()
    plan() if a.condition == "plan" else evaluate(a)
