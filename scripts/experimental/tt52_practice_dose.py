"""Compare shorter and longer carried practice under one new stream of continuous games."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt46_continuous_season as season
from . import tt48_practice_transfer as transfer
from .tt4_transfer_trial import _read, _write, _hash

ROOT = Path("artifacts/tt52_practice_dose")
PLAN = ROOT / "schedule.json"
course = season.course


def freeze():
    cards = _read(transfer.PLAN)["practice"]
    pool = {c["id"]: c for c in cards}
    if len(pool) != 32:
        raise ValueError("Expected the full untouched release pool")
    exclusions = {c["id"] for c in _read(season.CARRY)["cards"][:48]}
    exclusions.update(r["id"] for r in _read(season.fresh.RISK)["rows"])
    for source in season.fresh.servo.SOURCES:
        exclusions.update(
            r["card"]["id"] for r in _read(source.with_suffix(".json"))["practice"]
        )
    if exclusions & set(pool):
        raise ValueError("Prior material contains a test release")
    rng = np.random.default_rng(52074863)
    stream = []
    for _ in range(3):
        for i in rng.permutation(sorted(pool)):
            stream.append(
                dict(
                    pool[i],
                    seed=int(rng.integers(2**30)),
                    reply_seed=int(rng.integers(2**30)),
                )
            )
    ROOT.mkdir(parents=True, exist_ok=True)
    _write(
        PLAN,
        dict(
            practice=stream,
            source=_hash(transfer.PLAN),
            seed=52074863,
            rule="Three blocks of 32 independent launch and reply streams. Start with 12 or 48 games of the same practice, keep updating within and across all games. All completed points count.",
        ),
    )


def run(a):
    a.carry = course.ROOT / f"learning-{a.games:02d}.json.gz"
    if len(_read(a.carry)["rows"]) != a.games:
        raise ValueError("Invalid dose")
    risk = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    with (
        patch.object(course, "PLAN", PLAN),
        patch.object(
            course.mpc, "MPC", lambda *x, **kw: season.Player(*x, coverage=risk, **kw)
        ),
    ):
        course.run(a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("games", type=int, choices=(0, 12, 48))
    p.add_argument("--n", type=int, default=96)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(cap=240, mode="live", deck="practice", resume=False)
    a = p.parse_args()
    a.out = ROOT / f"live-from-{a.games:02}.json.gz"
    freeze() if not a.games else run(a)
