"""Reconstruct the exact state before a long point and freeze within-point learning alone."""

import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np

from . import tt46_continuous_season as season
from . import tt47_season_retention as retention
from . import tt48_practice_transfer as transfer
from .tt40_within_rally import replay_memory
from .tt19_view_full_game import digest_json
from .tt4_transfer_trial import _hash, _read, _write

course, fresh = season.course, season.fresh
mpc = course.mpc


def run(source, index, output):
    d = _read(source)
    carry = transfer.SOURCE
    if (
        d["plan_sha256"] != _hash(transfer.PLAN)
        or d["carry_sha256"] != _hash(carry)
        or d["cards"][: index + 1] != _read(transfer.PLAN)["practice"][: index + 1]
        or d["mode"] != "live"
    ):
        raise ValueError("Unexpected source or measure")
    with np.load(mpc.BANK) as b:
        raw = mpc.DeepField(b, "local")
    with mpc.centered(d["ready"]):
        a = mpc.arena()
        sensors = mpc.resources(a)
        with patch.object(
            mpc,
            "Fragments",
            lambda f, e: course.Carried(
                f, e, practice=[course.PRACTICE], held={c["id"] for c in d["cards"]}
            ),
        ):
            actor = season.Player(
                mpc.BANK,
                raw,
                coverage=fresh.shot.Coverage(_read(fresh.RISK)),
                budget=d["budget"],
                variant=d["variant"],
            )
        try:
            retention.ancestral_restore(
                actor,
                carry.with_suffix("").with_suffix(".npz"),
                _read(carry),
                season.CARRY,
            )
            for i in range(index + 1):
                c, r = d["cards"][i], d["rows"][i]
                if actor.priors.digest() != r["memory"]["before"]:
                    raise ValueError("Incorrect pregame memory")
                if i < index:
                    actor.begin()
                    replay_memory(
                        a,
                        sensors,
                        actor.priors,
                        c,
                        r,
                        d["ready"],
                        course.OPTIC,
                        actor.serial,
                    )
                    continue
                f = mpc.play(
                    a,
                    *sensors,
                    d["ready"],
                    c,
                    r["shot"],
                    raw,
                    actor,
                    course.OPTIC,
                    "frozen",
                    d["cap"],
                    20,
                )
                first = next(
                    (
                        k
                        for k, (u, v) in enumerate(zip(r["actions"], f["actions"]))
                        if u != v
                    ),
                    None,
                )
                if first is not None and digest_json(
                    [e for e in r["events"] if e["step"] <= first]
                ) != digest_json([e for e in f["events"] if e["step"] <= first]):
                    raise ValueError(
                        "Physical discrepancy before the first different action"
                    )
                if first is None and digest_json(r["events"]) != digest_json(
                    f["events"]
                ):
                    raise ValueError("Unequal physics with the same actions")
                if f["memory"]["after"] != r["memory"]["before"]:
                    raise ValueError("Frozen memory changed")
                _write(
                    output,
                    dict(
                        source_sha256=_hash(source),
                        card=c,
                        index=i,
                        live=r["score"]["learner"],
                        frozen=f["score"]["learner"],
                        first_changed_command=first,
                        first_changed_at_score=sum(
                            e["side"] == 1
                            for e in r["events"]
                            if e["kind"] == "legal_return"
                            and first is not None
                            and e["step"] <= first
                        )
                        - 1,
                        ready=d["ready"],
                        row=f,
                    ),
                )
        finally:
            actor.close()
            sensors[0].close()
            a.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("index", type=int)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    run(args.source, args.index, args.output)
