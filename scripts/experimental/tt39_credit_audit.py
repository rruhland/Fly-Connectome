"""Compare camera-selected training windows against scored impacts after play."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write


def audit(bank, out):
    doc = _read(bank.with_suffix(".json"))
    with np.load(bank) as z:
        marks = z["marks"]
    if _hash(bank) != doc["bank_sha256"]:
        raise ValueError("Changed observation bank")
    src = {name: _read(bank.with_name(name)) for name in doc["sources"]}
    for name in src:
        if _hash(bank.with_name(name)) != doc["sources"][name]:
            raise ValueError("Changed action tape")
    ptr = 0
    distances = []
    missing = 0
    count = 0
    for r in doc["practice"]:
        d = src[r["source"]]
        ix = r["source_row"]
        if r["card"] != d["cards"][ix]:
            raise ValueError("Incorrect tape row")
        m = marks[ptr : ptr + r["frames"]]
        ptr += len(m)
        centers = np.unique(np.round(m[:, 0] / 60 - m[:, 1], 6))
        actual = np.asarray(
            [s["hit"] for s in d["rows"][ix]["score"]["shots"] if s["side"] == 1]
        )
        if len(centers) != r["mentor_returns"] or not len(actual):
            raise ValueError("Miscount in camera credit receipt")
        idx = np.array([np.argmin(abs(actual - k)) for k in centers])
        missing += len(actual) - len(np.unique(idx))
        errors = np.abs(actual[idx] - centers)
        count += len(actual)
        distances.extend(errors.tolist())
    if ptr != len(marks):
        raise ValueError("Trailing camera annotations")
    _write(
        out,
        {
            "scored_contacts_in_recorded_points": count,
            "visual_credits": len(distances),
            "omitted": missing,
            "within_50ms": sum(t <= 0.050 for t in distances),
            "within_100ms": sum(t <= 0.100 for t in distances),
            "time_error_quantiles_s": np.quantile(distances, [0, 0.5, 0.9, 1]).tolist(),
            "credit_error_s": distances,
            "source_sha256": _hash(bank),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bank", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    audit(a.bank, a.out)
