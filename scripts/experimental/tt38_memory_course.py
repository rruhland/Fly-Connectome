"""Select independent random streams and initial physical recordings for motor-memory transfer."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import measured_launches
from .tt31_curriculum import card, opening_candidates


def prepare(out, feeds, priors):
    used = set()
    for path in priors:
        doc = _read(path)
        used.update(doc.get("exclusions", []))
        for split in ("practice", "held_openings", "held_replies", "cards"):
            used.update(int(row["id"]) for row in doc.get(split, []))
    pool = [
        s["id"]
        for s in opening_candidates(measured_launches(feeds))
        if s["id"] not in used
    ][:600]
    rng = np.random.default_rng(3812465)
    ids = rng.choice(pool, 48, replace=False)
    rows = [
        dict(
            card(int(s), "full", i, int(rng.integers(2**30))),
            opening="random",
            profile="reach",
            reply_seed=int(rng.integers(2**30)),
        )
        for i, s in enumerate(ids)
    ]
    _write(
        out,
        {
            "practice": rows[:24],
            "replication": rows[24:],
            "feed_sha256": _hash(feeds),
            "excluded": {p.name: _hash(p) for p in priors},
            "criterion": "24 consecutive open-loop assignments to each policy, 24 additional for independent replication. Entire point physical with fresh draws from the reactive far player.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("prior", nargs="+", type=Path)
    parser.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    a = parser.parse_args()
    prepare(a.out, a.feeds, a.prior)
