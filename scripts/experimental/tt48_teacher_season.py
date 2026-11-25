"""Combine the small earned camera demonstration bank with continuously adapted joint feedback."""

import argparse
from pathlib import Path
from unittest.mock import patch

from . import tt42_learning_course as course
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write
from .tt42_commit_teacher_student import FRAGMENTS, OUT, WithTeacher

PILOT = OUT / "earned-camera.npz"


def run(a):
    bank = _read(PILOT.with_suffix(".json"))
    if bank["bank_sha256"] != _hash(PILOT):
        raise ValueError("Changed teacher demonstration bank")
    teacher_releases = {r["card"]["id"] for r in bank["practice"]}

    def make(file, engine, *, held, practice):
        if practice != [course.PRACTICE] or held & teacher_releases:
            raise ValueError(
                "Incompatible practice archive or overlapping teacher release"
            )
        return WithTeacher(file, engine, FRAGMENTS(PILOT, engine), held=held, use=True)

    with patch.object(course, "Carried", make):
        season.run(a)
    d = _read(a.out)
    d["teacher_sha256"] = _hash(PILOT)
    _write(a.out, d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    parser.set_defaults(
        deck="practice", mode="live", carry=season.CARRY, cap=240, resume=False
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run(args)
