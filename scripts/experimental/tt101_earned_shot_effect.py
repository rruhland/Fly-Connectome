"""What changed immediately after the first shared ball both motor memories actually returned?"""

import argparse
from pathlib import Path

import numpy as np

from . import tt51_afterstate as incoming
from . import tt73_partner_rehearsal as school
from . import tt101_body_practice as confirm
from .tt4_transfer_trial import _hash, _read, _write


def effects(before, after):
    base = _read(before)
    adapted = _read(after)
    if (
        base["cards"] != adapted["cards"]
        or len(base["rows"]) != len(base["cards"])
        or len(adapted["rows"]) != len(base["cards"])
    ):
        raise ValueError("Unfinished or unpaired physics course")
    rows = []
    for card, a, b in zip(base["cards"], base["rows"], adapted["rows"], strict=True):
        da, db = (r["placements"]["delivered"] for r in (a, b))
        for j, (u, v) in enumerate(zip(da, db)):
            if u.get("far_contact") != v.get("far_contact") or u.get(
                "near_bounce"
            ) != v.get("near_bounce"):
                break
            if not (u["near_landed"] and v["near_landed"]):
                break
            # Compare the first actual changed outgoing stroke on a common arriving ball.
            if u["near_contact"] == v["near_contact"]:
                continue
            out = []
            if len(da) <= j + 1 or len(db) <= j + 1:
                break
            if (
                a["placements"]["requested"][j + 1]["target"]
                != b["placements"]["requested"][j + 1]["target"]
            ):
                raise ValueError("Different future independent request")
            for r, prior, next_offer in ((a, u, da[j + 1]), (b, v, db[j + 1])):
                when = prior["near_contact"]["time"]
                s = next(
                    e
                    for e in r["events"]
                    if e.get("collider") == "learner_racket"
                    and 0 < e["time"] - when < 0.1
                )
                ray = next_offer.get("receive_line")
                future = (
                    incoming.projected(np.r_[ray["xyz"], ray["velocity"]]).tolist()
                    if ray
                    else None
                )
                out.append(
                    dict(
                        velocity=s["velocity"],
                        spin=s["spin"],
                        future=future,
                        next_scored=next_offer["near_landed"],
                        total=r["score"]["learner"],
                    )
                )
            rows.append(dict(id=card["id"], leg=j, before=out[0], after=out[1]))
            break
    observed = [
        r for r in rows if all(r[k]["future"] is not None for k in ("before", "after"))
    ]
    diffs = {}
    for name in ("velocity", "spin", "future", "next_scored", "total"):
        delta = np.array(
            [
                np.asarray(r["after"][name], float)
                - np.asarray(r["before"][name], float)
                for r in observed
            ]
        )
        if not len(delta):
            continue
        draw = np.random.default_rng(101658).integers(
            len(delta), size=(10000, len(delta))
        )
        diffs[name] = dict(
            mean=delta.mean(0).tolist(),
            median=np.median(delta, 0).tolist(),
            interval=np.quantile(delta[draw].mean(1), [0.025, 0.975], axis=0).tolist(),
        )
    return dict(
        before=_hash(before),
        after=_hash(after),
        all_games=len(base["cards"]),
        first_different_dual_score=len(rows),
        both_following_seen=len(observed),
        paired_deltas=diffs,
        rows=rows,
    )


def run(d):
    if d == "original-held":
        a = confirm.restore.ROOT / "original-before.json.gz"
        b = school.ROOT / "course/uniform-original.json.gz"
    else:
        a = confirm.ROOT / "before-held.json.gz"
        b = confirm.ROOT / "lateral-held.json.gz"
    out = effects(a, b)
    _write(confirm.ROOT / (d + "-shot-effects.json"), out)
    print(out["both_following_seen"], out["paired_deltas"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cohort", choices=("original-held", "replication"))
    run(p.parse_args().cohort)
