"""Test the learned memory from immediately before the 63-return point on independent starts."""

import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np

from . import tt51_transfer_after_point as previous
from . import tt101_body_practice as cohort
from .tt4_transfer_trial import _hash, _read, _write
from .tt54_training_speedups import enabled

ROOT = Path("artifacts/tt107_continued_interception")
PLAN = ROOT / "protocol.json"
CARRY = previous.ROOT / "after-09.json.gz"
course, season = previous.course, previous.season


def plan():
    d = _read(CARRY)
    if d["exact_source"] != _hash(previous.SOURCE) or d["checkpoint_sha256"] != _hash(
        CARRY.with_suffix("").with_suffix(".npz")
    ):
        raise ValueError("Starting memory differs from independent reconstruction")
    starting = _read(cohort.PLAN)["cards"]
    seen = set()
    for p in (CARRY, previous.stream.SOURCE, season.CARRY):
        training = _read(p)
        seen.update(c["id"] for c in training["cards"][: len(training["rows"])])
    if seen & {c["id"] for c in starting}:
        raise ValueError("Earlier motor practice in evaluation")
    rng = np.random.default_rng(10710258)
    course_cards = starting.copy()
    for _ in range(3):
        course_cards.extend(
            dict(
                starting[i],
                seed=int(rng.integers(1, 2**30)),
                reply_seed=int(rng.integers(1, 2**30)),
            )
            for i in rng.permutation(len(starting))
        )
    ROOT.mkdir(exist_ok=True)
    _write(
        PLAN,
        dict(
            practice=course_cards,
            carry=_hash(CARRY),
            bank=d["checkpoint_sha256"],
            training_disjoint=True,
            training_releases=sorted(seen),
            cohort=_hash(cohort.PLAN),
            rule="Continue the actual earned point-boundary memory that once produced 63. First 24 independent measured releases match the new TT101 assessment exactly. The later 72 points reshuffle the same releases with new continuous draws, without resetting achieved learning at any boundary. One physical launch each. A frozen first block isolates the carried ability from new updates.",
        ),
    )


def run(a):
    design = _read(PLAN)
    if design["carry"] != _hash(CARRY) or design["cohort"] != _hash(cohort.PLAN):
        raise ValueError("Course or prior differs")
    rank = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    ordinary = course.resume

    def restore(model, path, d):
        if path == CARRY.with_suffix("").with_suffix(".npz"):
            d = dict(d, rows=[None] * d["parent_serial"] + d["rows"])
        return ordinary(model, path, d)

    with (
        enabled(),
        patch.object(course, "PLAN", PLAN),
        patch.object(course, "resume", restore),
        patch.object(
            course.mpc, "MPC", lambda *v, **kw: season.Player(*v, coverage=rank, **kw)
        ),
    ):
        course.run(a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "live", "frozen"))
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--resume", action="store_true")
    p.set_defaults(carry=CARRY, deck="practice", cap=240)
    a = p.parse_args()
    a.n = 96 if a.mode == "live" else 24
    a.out = ROOT / (a.mode + ".json.gz")
    plan() if a.mode == "plan" else run(a)
