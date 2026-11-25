"""Freeze only within each measured season point, restoring the true learner before the next game."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt40_within_rally import replay_memory

mpc = course.mpc


def run(source, out):
    d = _read(source)
    held = {c["id"] for c in d["cards"]}
    initial = _read(season.CARRY)
    if d["plan_sha256"] != _hash(season.PLAN) or d["carry_sha256"] != _hash(
        season.CARRY
    ):
        raise ValueError("Unmatched season conditions")
    with np.load(mpc.BANK) as b:
        raw = mpc.DeepField(b, "local")
    with mpc.centered(d["ready"]):
        a = mpc.arena()
        sensors = mpc.resources(a)
        with patch.object(
            mpc,
            "Fragments",
            lambda f, e: course.Carried(f, e, practice=[course.PRACTICE], held=held),
        ):
            model = season.Player(
                mpc.BANK,
                raw,
                coverage=fresh.shot.Coverage(_read(fresh.RISK)),
                variant=d["variant"],
                budget=d["budget"],
            )
        course.resume(model, season.CARRY.with_suffix("").with_suffix(".npz"), initial)
        result = dict(
            source=_hash(source), initial=_hash(season.CARRY), rows=[], pairs=[]
        )
        try:
            for card, live in zip(d["cards"], d["rows"]):
                if model.priors.digest() != live["memory"]["before"]:
                    raise ValueError("Different opening memory")
                r = mpc.play(
                    a,
                    *sensors,
                    d["ready"],
                    card,
                    live["shot"],
                    raw,
                    model,
                    course.OPTIC,
                    "frozen",
                    d["cap"],
                )
                first = next(
                    (
                        k
                        for k, (x, y) in enumerate(zip(live["actions"], r["actions"]))
                        if x != y
                    ),
                    None,
                )
                if first is not None and digest_json(
                    [e for e in r["events"] if e["step"] <= first]
                ) != digest_json([e for e in live["events"] if e["step"] <= first]):
                    raise ValueError(
                        "Physics diverged before the first changed command"
                    )
                if first is None and (
                    r["actions"] != live["actions"]
                    or digest_json(r["events"]) != digest_json(live["events"])
                ):
                    raise ValueError("Same actions did not produce identical game")
                result["rows"].append(r)
                result["pairs"].append(
                    dict(
                        card=card["id"],
                        active=live["score"]["learner"],
                        frozen=r["score"]["learner"],
                        first_changed_command=first,
                    )
                )
                replay_memory(
                    a,
                    sensors,
                    model.priors,
                    card,
                    live,
                    d["ready"],
                    course.OPTIC,
                    model.serial,
                )
                if model.priors.digest() != live["memory"]["after"]:
                    raise ValueError("Original learning state was not reconstructed")
                _write(out, result)
                print(result["pairs"][-1], flush=True)
        finally:
            model.close()
            sensors[0].close()
            a.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    run(a.source, a.out)
