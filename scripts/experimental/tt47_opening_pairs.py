"""Test opening-ball coverage on exactly matched physical deliveries across a carried course."""

import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json


def compare(left, right, out):
    a, b = _read(left), _read(right)
    if (
        a["cards"] != b["cards"]
        or len(a["rows"]) != len(a["cards"])
        or len(b["rows"]) != len(a["cards"])
    ):
        raise ValueError("Unmatched or incomplete assignment")
    pairs = []
    for c, r, s in zip(a["cards"], a["rows"], b["rows"], strict=True):
        offered = [
            next((z for z in k["placements"]["delivered"] if z["near_bounce"]), None)
            for k in (r, s)
        ]
        if offered[0] is None and offered[1] is None:
            pairs.append(dict(id=c["id"], delivered=False))
            continue
        if any(z is None for z in offered):
            raise ValueError("Only one arm received an opening")
        for field in ("far_contact", "near_bounce"):
            if digest_json(offered[0][field]) != digest_json(offered[1][field]):
                raise ValueError(f"First native {field} did not match")
        pairs.append(
            dict(
                id=c["id"],
                delivered=True,
                control=offered[0]["near_landed"],
                candidate=offered[1]["near_landed"],
                bounce_sha256=digest_json(offered[0]["near_bounce"]),
            )
        )
    milestones = {
        str(k): [sum(r["score"]["learner"] >= k for r in d["rows"]) for d in (a, b)]
        for k in [1, 2, 5, 10, 15, 20, 30, 50, 100]
    }
    _write(
        out,
        dict(
            control=_hash(left),
            candidate=_hash(right),
            assignments=pairs,
            games_reaching=milestones,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for s in ("control", "candidate", "out"):
        p.add_argument(s, type=Path)
    a = p.parse_args()
    compare(a.control, a.candidate, a.out)
