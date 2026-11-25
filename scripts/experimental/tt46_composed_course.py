"""Carry the two-stage direct joint search through matched games with the original broad opponent."""

import argparse
from pathlib import Path
from unittest.mock import patch

from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from . import tt45_composed_feedback as motion
from .tt4_transfer_trial import _hash, _read, _write
from .tt45_rollout_reuse import ReuseMPC


class Player(motion.Composed, fresh.Hybrid, ReuseMPC):
    pass


def run(a):
    model = fresh.shot.Coverage(_read(fresh.RISK))
    ordinary = course.mpc.play

    def play(*args, **kwargs):
        result = ordinary(*args, **kwargs)
        result["coverage_edits"] = args[10].reranks
        return result

    def build(*args, **kwargs):
        return Player(*args, coverage=model, **kwargs)

    with (
        patch.object(course, "PLAN", fresh.PLAN),
        patch.object(course.mpc, "MPC", build),
        patch.object(course.mpc, "play", play),
    ):
        course.run(a)
    d = _read(a.out)
    d.update(
        composition_sha256=_hash(Path(motion.__file__)),
        risk=_hash(fresh.RISK),
        temporal=_hash(fresh.servo.ROOT / "prior.npz"),
        schedule=_hash(fresh.PLAN),
        budget_note="Both use 28 ordinary candidate slots. The composed policy additionally screens second-stage motions after a weak decision.",
    )
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        deck="assessment", mode="live", carry=fresh.CARRY, cap=240, resume=False
    )
    run(p.parse_args())
