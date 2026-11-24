"""Summarize intact physical points and independently replay the longest cases."""

import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt37_joint_report import replay_case


def report(paths, target, verify):
    runs = {}
    for path in paths:
        doc = _read(path)
        if len(doc["rows"]) != len(doc["cards"]):
            raise ValueError(f"Incomplete {path}")
        counts = [row["score"]["learner"] for row in doc["rows"]]
        i = max(
            range(len(counts)),
            key=lambda ix: (counts[ix], doc["rows"][ix]["score"]["returns"]),
        )
        totals = {
            "points": len(counts),
            "offered": sum(bool(row["placements"]["delivered"]) for row in doc["rows"]),
            "own": sum(counts),
            "later": sum(max(0, c - 1) for c in counts),
            "own_by_point": counts,
            "best_own": counts[i],
            "best_alternating": doc["rows"][i]["score"]["returns"],
            "sha256": _hash(path),
            "mode": doc["mode"],
        }
        if verify:
            totals["best_replay"] = replay_case(
                doc["ready"], "random", doc["cards"][i], doc["rows"][i]
            )
        runs[path.name] = totals
    _write(target, runs)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", type=Path)
    p.add_argument("paths", type=Path, nargs="+")
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    report(a.paths, a.target, a.verify)
