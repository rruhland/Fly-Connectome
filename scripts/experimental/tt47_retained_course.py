"""Carry coverage-preserving motor memory into a full course with the best paired feedback combination."""

import argparse
from pathlib import Path
from unittest.mock import patch

from . import tt42_diverse_replay as replay
from . import tt42_learning_course as course
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write

CARRY = course.ROOT / "diverse-learning-48.json.gz"
ROOT = Path("artifacts/tt47_retention")


def run(a):
    with patch.object(course, "Carried", replay.Diverse):
        season.run(a)
    d = _read(a.out)
    d.update(
        repertoire_sha256=_hash(Path(replay.__file__)),
        initial_repertoire=_hash(a.carry),
    )
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("frozen", "live"))
    p.add_argument("out", type=Path)
    p.add_argument("--deck", choices=("practice", "assessment"), default="practice")
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--carry", type=Path, default=CARRY)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(cap=240, resume=False)
    a = p.parse_args()
    a.out.parent.mkdir(exist_ok=True, parents=True)
    run(a)
