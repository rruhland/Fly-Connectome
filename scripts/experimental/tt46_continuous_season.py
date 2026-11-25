"""Measure sustained return learning over a full season with fresh continuous requests every game."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from .tt4_transfer_trial import _hash, _read, _write
from .tt45_rollout_reuse import ReuseMPC

ROOT = Path("artifacts/tt46_season")
PLAN = ROOT / "schedule.json"
CARRY = course.ROOT / "learning-48.json.gz"


def plan():
    # Measured start diversity and self-contained independent reply streams.
    # Cross-fitting is by release ID, before any schedule outcomes exist.
    used = {
        c["id"] for name in ("practice", "assessment") for c in _read(course.PLAN)[name]
    }
    used.update(c["id"] for c in _read(fresh.PLAN)["assessment"])
    used.update(r["id"] for r in _read(fresh.RISK)["rows"])
    for p in fresh.servo.SOURCES:
        used.update(r["card"]["id"] for r in _read(p.with_suffix(".json"))["practice"])
    pool = {c["id"]: c for c in _read(course.SOURCE)["practice"] if c["id"] not in used}
    if len(pool) < 16:
        raise ValueError("Too few independent opening recordings")
    rng = np.random.default_rng(460574)
    ids = rng.permutation(list(pool))
    practice, assessment = [], []
    for block in range(8):
        for i in rng.permutation(ids[:-8]):
            c = pool[i]
            practice.append(
                dict(
                    c,
                    seed=int(rng.integers(1, 2**30)),
                    reply_seed=int(rng.integers(1, 2**30)),
                )
            )
    for cycle in range(3):
        for i in rng.permutation(ids[-8:]):
            assessment.append(
                dict(
                    pool[i],
                    seed=int(rng.integers(1, 2**30)),
                    reply_seed=int(rng.integers(1, 2**30)),
                )
            )
    ROOT.mkdir(exist_ok=True, parents=True)
    _write(
        PLAN,
        dict(
            practice=practice,
            assessment=assessment,
            distinct=len(ids),
            earlier=_hash(fresh.PLAN),
            starting_checkpoint=_hash(CARRY),
            random_seed=460574,
            rule="All unused measured releases from the endurance pool after excluding both trained readouts and the previous direct and fresh courses. Eight shuffled blocks of practice. The last eight measured openings are reserved for independent assessment, with three independent opponent streams per release. Roster fixed before play. All physical losses count.",
        ),
    )


class Player(fresh.Hybrid, ReuseMPC):
    pass


def run(a):
    packet = _read(PLAN)
    if packet["starting_checkpoint"] != _hash(CARRY) or a.n > len(packet[a.deck]):
        raise ValueError("Changed starting state or incomplete season schedule")
    ranker = fresh.shot.Coverage(_read(fresh.RISK))

    def make(*args, **kw):
        return Player(*args, coverage=ranker, **kw)

    with patch.object(course, "PLAN", PLAN), patch.object(course.mpc, "MPC", make):
        course.run(a)
    result = _read(a.out)
    result.update(
        temporal=_hash(fresh.servo.ROOT / "prior.npz"),
        risk=_hash(fresh.RISK),
        schedule=_hash(PLAN),
    )
    _write(a.out, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "live", "frozen"))
    p.add_argument("out", type=Path, nargs="?")
    p.add_argument("--deck", default="practice", choices=("practice", "assessment"))
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--carry", type=Path, default=CARRY)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--resume", action="store_true")
    p.set_defaults(cap=240)
    args = p.parse_args()
    plan() if args.mode == "plan" else run(args)
