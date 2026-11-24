"""Carry the exact post-game reward predictor into a disjoint course."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write


def checkpoint(source, target):
    d = _read(source)
    if not d["rows"] or len(d["rows"]) != len(d["cards"]):
        raise ValueError("A complete course is required")
    count = 4 if d.get("rally_value") else 2
    prior = d.get("motor_value")
    if prior and prior != "None":
        prior = Path(prior)
        base = _read(prior)
        prior_hash = _hash(prior)
    else:
        base = {"x": [np.zeros(11).tolist()], "y": [[0.5] * count]}
        prior_hash = None
    added = d["value_experience"][-512:]
    if len(d["value_experience"]) != d["rows"][-1]["value_examples_so_far"]:
        raise ValueError("Incomplete reward history")
    x = base["x"] + [r[0] for r in added]
    y = base["y"] + [r[1] for r in added]
    _write(
        target,
        {
            "x": x,
            "y": y,
            "n": len(x),
            "prior_sha256": prior_hash,
            "source_sha256": _hash(source),
            "source": source.name,
            "practice_ids": sorted(
                set(base.get("practice_ids", base.get("release", [])))
                | {c["id"] for c in d["cards"]}
            ),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("target", type=Path)
    args = p.parse_args()
    checkpoint(args.source, args.target)
