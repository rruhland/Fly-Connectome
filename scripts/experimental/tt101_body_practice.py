"""Is the cross-body practice gain reproducible and does changing clamps add value?"""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt70_body_retention as restore
from . import tt73_partner_rehearsal as teacher
from . import tt97_return_to_center as transfer
from .tt4_transfer_trial import _hash, _read, _write
from .tt31_curriculum import card, opening_candidates

ROOT = Path("artifacts/tt101_body_practice")
PLAN = ROOT / "assessment.json"
PRACTICE = ROOT / "uniform.json.gz"
season, course = transfer.season, transfer.course


def plan(feeds):
    original = _read(restore.PLAN)
    used = {c["id"] for c in original["cards"]} | set(
        _read(transfer.PLAN)["carried_ids"]
    )
    for path in set(season.fresh.servo.SOURCES) | {course.mpc.BANK, course.PRACTICE}:
        used.update(
            r["card"]["id"] for r in _read(path.with_suffix(".json"))["practice"]
        )
    used.update(r["id"] for r in _read(season.fresh.RISK)["rows"])
    # Withhold every launch of this specific matched practice experiment.
    used.update(c["id"] for c in _read(teacher.ROOT / "course/course.json")["cards"])
    pool = [
        s["id"]
        for s in opening_candidates(course.mpc.measured_launches(feeds))
        if s["id"] not in used
    ]
    if len(set(pool)) != len(pool) or len(pool) < 24:
        raise ValueError("Too few distinct eligible release recordings")
    rng = np.random.default_rng(101483025)
    cards = [
        dict(
            card(int(ident), "full", j, int(rng.integers(1, 2**30))),
            profile="reach",
            opening="random",
            reply_seed=int(rng.integers(1, 2**30)),
        )
        for j, ident in enumerate(rng.choice(pool, 24, replace=False))
    ]
    ROOT.mkdir(exist_ok=True)
    _write(
        PLAN,
        dict(
            original,
            cards=cards,
            eligible=len(pool),
            launch_feed=_hash(feeds),
            forbidden=sorted(used),
            training=_hash(teacher.ROOT / "course/course.json"),
            design="24 new measured launch recordings and wholly fresh independent random opponent draws, assessed on the original centered arm. Paired frozen copies before practice, after ordinary shifted-clamp practice, and after an equal update dose of ordinary centered practice. Same fixed order of potential training cards, 256 offered-ball learning limit, then finish current active point. All held launches count.",
        ),
    )


def train(feeds):
    ROOT.mkdir(exist_ok=True)
    with (
        patch.object(teacher.prior, "ROOT", ROOT),
        patch.object(teacher.prior, "PLAN", teacher.ROOT / "course/course.json"),
    ):
        teacher.prior.practice("uniform", feeds, teacher.Coach, centered=True)


def assess(which, feeds):
    src = {"before": None, "lateral": transfer.PRACTICE, "centered": PRACTICE}[which]
    if src and not _read(src)["complete"]:
        raise ValueError("Incomplete learning boundary")
    with patch.object(restore, "PLAN", PLAN):
        restore.run(
            "original",
            "before" if src is None else "after",
            feeds,
            acquired=src,
            destination=ROOT / (which + "-held.json.gz"),
        )


def report():
    from .tt53_responsive_report import compare

    centers = {
        k: ROOT / (k + "-held.json.gz") for k in ("before", "lateral", "centered")
    }
    out = {}
    for k, others in [("lateral", ("before", "centered")), ("centered", ("before",))]:
        r = compare(dict(responsive=centers[k], **{j: centers[j] for j in others}), 24)
        r["groups"][k] = r["groups"].pop("responsive")
        out[k] = r
    _write(ROOT / "results.json", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=("plan", "train", "before", "lateral", "centered", "report")
    )
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    a = p.parse_args()
    if a.stage == "plan":
        plan(a.feeds)
    elif a.stage == "train":
        train(a.feeds)
    elif a.stage == "report":
        report()
    else:
        assess(a.stage, a.feeds)
