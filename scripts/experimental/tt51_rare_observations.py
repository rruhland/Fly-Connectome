"""Enrich practice with real older outgoing situations that previously yielded a low return."""

import argparse
from pathlib import Path
import numpy as np
from .tt4_transfer_trial import _read, _write
from .tt39_opportunity_map import projected_height
from .tt50_opponent_vision import ROOT, course, collect, SRC, roster

PLAN = ROOT / "difficult-plan.json"


def schedule():
    base = _read(SRC)
    reserved = {base["cards"][i]["id"] for i in roster(base)[16:]}
    options = {}
    for p in (
        SRC,
        course.ROOT / "split-memory-48.json.gz",
        course.ROOT / "diverse-learning-48.json.gz",
    ):
        d = _read(p)
        for i, (c, r) in enumerate(zip(d["cards"], d["rows"])):
            if c["id"] in reserved:
                continue
            for leg, offer in enumerate(r["placements"]["delivered"]):
                if (
                    not leg
                    or not offer.get("receive_line")
                    or not offer.get("far_contact")
                ):
                    continue
                h = projected_height(offer["receive_line"])
                if h > 0.997:
                    continue
                far = offer["far_contact"]["time"]
                key = (
                    c["id"],
                    round(far, 6),
                    tuple(np.round(offer["receive_line"]["xyz"], 5)),
                )
                options.setdefault(
                    key, dict(id=c["id"], source=str(p), index=i, leg=leg, old_height=h)
                )
    rng = np.random.default_rng(510841)
    cases = list(options.values())
    ids = np.sort(list(set(c["id"] for c in cases)))
    rng.shuffle(ids)
    held = set(ids[-5:])
    rng.shuffle(cases)
    cases.sort(key=lambda c: c["id"] in held)
    for c in cases:
        c["split"] = "held" if c["id"] in held else "practice"
    _write(
        PLAN,
        dict(
            cases=cases,
            ordinary_test_reserved=sorted(reserved),
            rule="Archived physical receiving height below 0.997 m selects the practice situation. One original and five independently redrawn replies per outgoing state. Targets for learning come exclusively from subsequent RGB. All points sharing a release share a split.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("start", type=int)
    p.add_argument("stride", type=int)
    args = p.parse_args()
    if not PLAN.exists():
        schedule()
    cases = _read(PLAN)["cases"]
    collect(range(args.start, len(cases), args.stride), cases)
