"""An independent continuing stream and reward-frozen counterfactual for actual long rallies."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt52_practice_dose as dose
from .tt4_transfer_trial import _hash, _read, _write
from .tt54_training_speedups import enabled

ROOT = Path("artifacts/tt80_continuous_games")
PLAN = ROOT / "protocol.json"
season, course = dose.season, dose.course


def freeze():
    pool = {c["id"]: c for c in _read(dose.PLAN)["practice"]}
    source = _read(season.CARRY)
    earlier = {c["id"] for c in source["cards"][: len(source["rows"])]}
    if earlier & set(pool) or len(pool) != 32:
        raise ValueError("New roster requires separate start recordings")
    # A new random stream per launch, four rounds with the entire roster.
    rng = np.random.default_rng(805327169)
    practice = []
    for _ in range(4):
        for i in rng.permutation(sorted(pool)):
            practice.append(
                dict(
                    pool[i],
                    seed=int(rng.integers(2**30)),
                    reply_seed=int(rng.integers(2**30)),
                )
            )
    ROOT.mkdir(exist_ok=True)
    data = dict(
        practice=practice,
        carry=_hash(season.CARRY),
        reserve_parent=_hash(dose.PLAN),
        standard="Original centered arm, independent continuous profile, physical partner. These 128 complete points compare one live sequence carrying all experience against the same initial checkpoint frozen for the first 32 fresh streams. Score every assigned point. Compare whole-course coverage, legal opportunities, actual threshold survival, cost and first/last 32. Candidate algorithms use the identical streams. No restarts or selection inside a game.",
    )
    if PLAN.exists() and _read(PLAN) != data:
        raise ValueError("Existing preregistration differs")
    _write(PLAN, data)


def run(a):
    design = _read(PLAN)
    if design["carry"] != _hash(season.CARRY) or design["reserve_parent"] != _hash(
        dose.PLAN
    ):
        raise ValueError("Changed source or course")
    rank = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    with (
        enabled(),
        patch.object(course, "PLAN", PLAN),
        patch.object(
            course.mpc,
            "MPC",
            lambda *args, **kw: season.Player(*args, coverage=rank, **kw),
        ),
    ):
        course.run(a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "live", "frozen"))
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--resume", action="store_true")
    p.set_defaults(carry=season.CARRY, cap=240, deck="practice")
    a = p.parse_args()
    a.n = 128 if a.mode == "live" else 32
    a.out = ROOT / (a.mode + ".json.gz")
    freeze() if a.mode == "plan" else run(a)
