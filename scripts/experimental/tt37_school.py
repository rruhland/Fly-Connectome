"""Preselect new measured practice releases and a disjoint confirmation course."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import measured_launches
from .tt31_curriculum import card, opening_candidates


def prepare(feeds, older, output):
    prior = _read(older)
    used = set(prior["exclusions"])
    for split in ("practice", "held_openings", "held_replies"):
        used.update(c["id"] for c in prior[split])
    rng = np.random.default_rng(374853)
    remaining = [
        s["id"]
        for s in opening_candidates(measured_launches(feeds))
        if s["id"] not in used
    ][:600]
    ids = rng.choice(remaining, 144, replace=False)
    roster = [
        card(int(s), "full", i, int(rng.integers(2**30)))
        | {
            "opening": "random",
            "profile": "reach",
            "reply_seed": int(rng.integers(2**30)),
        }
        for i, s in enumerate(ids)
    ]
    _write(
        output,
        {
            "practice": roster[:96],
            "held_openings": roster[96:],
            "feed_sha256": _hash(feeds),
            "previous_roster": _hash(older),
            "description": "144 distinct initial recordings; all first and later target draws random. The first 96 are training, the last 48 are reserved for an independent confirmation.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument(
        "--previous", type=Path, default=Path("artifacts/tt37/opponent-plan.json")
    )
    a = p.parse_args()
    prepare(a.feeds, a.previous, a.out)
