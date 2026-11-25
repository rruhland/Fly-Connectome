"""Prospective unknown releases for feedback from image history and predicted consequences of outgoing movement."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt43_outgoing_coverage as shot
from . import tt43_temporal_servo as servo
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt44_combined")
PLAN = ROOT / "fresh.json"
RISK = shot.ROOT / "outgoing-risk.json"
CARRY = course.ROOT / "learning-12.json.gz"


def protocol():
    trained = set()
    files = [p.with_suffix(".json") for p in servo.SOURCES]
    for p in files:
        trained.update(r["card"]["id"] for r in _read(p)["practice"])
    trained.update(r["id"] for r in _read(RISK)["rows"])
    earlier = _read(course.PLAN)
    trained.update(
        c["id"] for group in ("practice", "assessment") for c in earlier[group]
    )
    source = _read(course.SOURCE)
    eligible = [c for c in source["practice"] if c["id"] not in trained]
    rng = np.random.default_rng(44182365)
    cards = [
        c | {"seed": int(rng.integers(2**30)), "reply_seed": int(rng.integers(2**30))}
        for c in eligible[:24]
    ]
    if len({c["id"] for c in cards}) != 24:
        raise ValueError("Insufficient distinct new releases")
    ROOT.mkdir(exist_ok=True, parents=True)
    _write(
        PLAN,
        {
            "practice": [],
            "assessment": cards,
            "trained_release_count": len(trained),
            "sources": {
                str(p): _hash(p)
                for p in [
                    RISK,
                    course.PLAN,
                    course.SOURCE,
                    *files,
                    CARRY,
                    servo.ROOT / "prior.npz",
                ]
            },
            "rule": "First 24 unused measured releases after removing every practice release in either fitted readout, all earlier prospective practice and assessment releases. Newly drawn opening and reply seeds, standard broad physical opponent throughout. One motor bank starts at the 12-game boundary and learns during the full run.",
        },
    )


class Hybrid(servo.Temporal, shot.Actor):
    pass


def run(a):
    d = _read(PLAN)
    if any(_hash(Path(p)) != h for p, h in d["sources"].items()):
        raise ValueError("Checkpoint changed")
    if a.actor == "stock":
        actor = course.mpc.MPC
    elif a.actor == "history":
        actor = servo.Temporal
    else:
        risk = shot.Coverage(_read(RISK))
        controller = shot.Actor if a.actor == "risk" else Hybrid

        def actor(*args, **kwargs):
            return controller(*args, **kwargs, coverage=risk)
    ordinary = course.mpc.play

    def play(*args, **kwargs):
        r = ordinary(*args, **kwargs)
        r["coverage_edits"] = getattr(args[10], "reranks", [])
        return r

    with (
        patch.object(course, "PLAN", PLAN),
        patch.object(course.mpc, "MPC", actor),
        patch.object(course.mpc, "play", play),
    ):
        course.run(a)
    saved = _read(a.out)
    saved["actor"] = a.actor
    saved["input_manifest"] = _hash(PLAN)
    _write(a.out, saved)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("actor", choices=("plan", "stock", "history", "risk", "hybrid"))
    p.add_argument("out", type=Path, nargs="?")
    p.add_argument("--mode", default="live", choices=("live", "frozen"))
    p.add_argument("--n", default=24, type=int)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(deck="assessment", cap=240, resume=False, carry=CARRY)
    a = p.parse_args()
    protocol() if a.actor == "plan" else run(a)
