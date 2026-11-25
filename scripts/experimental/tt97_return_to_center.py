"""Return to the original body after ordinary unassisted practice on a translated clamp."""

import argparse
from pathlib import Path
from unittest.mock import patch

from . import tt73_partner_rehearsal as coach
from . import tt80_prospective_course as games
from .tt4_transfer_trial import _hash, _read, _write
from .tt54_training_speedups import enabled

ROOT = Path("artifacts/tt97_return_to_center")
PRACTICE = coach.ROOT / "course/uniform.json.gz"
PLAN = ROOT / "protocol.json"
season, course = games.season, games.course


def freeze():
    src = _read(PRACTICE)
    course73 = _read(coach.ROOT / "course/course.json")
    if (
        not src["complete"]
        or src["fixture_dx_m"] != 0.12
        or course73["base"] != _hash(season.CARRY)
    ):
        raise ValueError("Wrong practice condition")
    baseline = season.ROOT / "learning.json.gz"
    res = _read(baseline)
    original = _read(season.PLAN)["practice"]
    ids = {c["id"] for c in src["cards"][: len(src["rows"])]}
    first = _read(season.CARRY)
    ids.update(c["id"] for c in first["cards"][: len(first["rows"])])
    if (
        ids & {c["id"] for c in original}
        or res["cards"] != original
        or len(res["rows"]) != len(original)
    ):
        raise ValueError("Previously trained release or unmatched control")
    ROOT.mkdir(exist_ok=True)
    _write(
        PLAN,
        dict(
            practice=original,
            bank=_hash(PRACTICE.with_suffix("").with_suffix(".npz")),
            source=_hash(PRACTICE),
            control_path=str(baseline),
            control=_hash(baseline),
            carried_ids=sorted(ids),
            standard="Start from the motor memory after 275 actual ordinary opponent deliveries practicing on the 12 cm lateral clamp. Return to the original centered fixture and continue learning for all 128 previously assigned ordinary season games. Same 16 launch recordings as the earlier frozen TT70 assessment, fresh draws disjoint from that assessment. Different releases from all motor practice. The original centered 48-game checkpoint played all of these 128 assignments prospectively before this transfer study. The contrast includes extra experience as well as a body change.",
        ),
    )


def run(a):
    protocol = _read(PLAN)
    if protocol["source"] != _hash(PRACTICE) or protocol["bank"] != _hash(
        PRACTICE.with_suffix("").with_suffix(".npz")
    ):
        raise ValueError("Starting experience differs")
    ordinary = course.resume

    def restore(model, path, d):
        if path == PRACTICE.with_suffix("").with_suffix(".npz"):
            d = dict(d, rows=[None] * len(_read(season.CARRY)["rows"]) + d["rows"])
        return ordinary(model, path, d)

    rank = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    with (
        enabled(),
        patch.object(course, "resume", restore),
        patch.object(course, "PLAN", PLAN),
        patch.object(
            course.mpc, "MPC", lambda *v, **kw: season.Player(*v, coverage=rank, **kw)
        ),
    ):
        course.run(a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("plan", "live"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        mode="live",
        deck="practice",
        n=128,
        carry=PRACTICE,
        cap=240,
        out=ROOT / "live.json.gz",
    )
    a = p.parse_args()
    freeze() if a.phase == "plan" else run(a)
