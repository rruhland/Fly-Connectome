"""Report point-level sustain and fixed-collection generalization with actual legal opportunities."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit


def totals(offered):
    return {
        "returned": sum(int(z["near_landed"]) for z in offered),
        "offered": len(offered),
    }


def summarize(d):
    s = []
    opportunities = []
    cards = d["cards"]
    if len(d["rows"]) != len(cards):
        raise ValueError("Unfinished course")
    for c, r in zip(cards, d["rows"], strict=True):
        for j, drawn in enumerate(r["placements"]["requested"]):
            target, kind = course.draw(c, j, "random", None)
            if target != tuple(drawn["target"]) or kind != drawn["draw"]:
                raise ValueError("Opponent draw differed")
        own = audit(r["events"], opening_side=1)["learner"]
        delivered = [z for z in r["placements"]["delivered"] if z.get("near_bounce")]
        if totals(delivered)["returned"] != own or own != r["score"]["learner"]:
            raise ValueError("Delivered ball status disagrees with native referee")
        s.append(own)
        opportunities.extend(delivered)
    heights = [
        (z, float(z["receive_line"]["xyz"][2]))
        for z in opportunities
        if z.get("receive_line")
    ]
    return {
        "points": len(s),
        "scores": s,
        "arm_returns": sum(s),
        "peak": max(s),
        "median": float(np.median(s)),
        "at_least_5": sum(v >= 5 for v in s),
        "at_least_10": sum(v >= 10 for v in s),
        "coverage": totals(opportunities),
        "leg": {
            key: totals([z for z in opportunities if lo <= z["leg"] < hi])
            for key, lo, hi in [
                ("opening", 0, 1),
                ("2_to_5", 1, 5),
                ("6_to_15", 5, 15),
                ("16_on", 15, 10000),
            ]
        },
        "receive_line_z": {
            key: totals([z for z, h in heights if lo <= h < hi])
            for key, lo, hi in [
                ("below_0.95", -100, 0.95),
                ("0.95_to_1.00", 0.95, 1.0),
                ("1.00_to_1.05", 1.0, 1.05),
                ("at_least_1.05", 1.05, 100),
            ]
        },
        "missing_line": len(opportunities) - len(heights),
        "full_point_cpu_s": sum(r["runtime_cpu_s"] for r in d["rows"]),
    }


def report(paths, out):
    reference = None
    result = {}
    names = Counter(path.name.split(".")[0] for path in paths)
    for path in paths:
        d = _read(path)
        if reference is None:
            reference = d["cards"]
        if reference != d["cards"]:
            raise ValueError("Unmatched evaluation cards")
        key = path.name.split(".")[0]
        if names[key] > 1:
            key = f"{path.parent.name}/{key}"
        if key in result:
            raise ValueError(f"Cannot distinguish result for {path}")
        result[key] = dict(source=_hash(path), **summarize(d))
    baseline = np.array(next(iter(result.values()))["scores"])
    idx = np.random.default_rng(43278).integers(
        0, len(baseline), (10000, len(baseline))
    )
    for row in result.values():
        diff = np.array(row["scores"]) - baseline
        row["paired"] = {
            "gain": int(diff.sum()),
            "improved": int((diff > 0).sum()),
            "declined": int((diff < 0).sum()),
            "unchanged": int((diff == 0).sum()),
            "mean_diff_interval": np.quantile(
                diff[idx].mean(1), [0.025, 0.975]
            ).tolist(),
        }
    _write(
        out,
        {
            "conditions": result,
            "uncertainty": "Resample assigned games as paired units; descriptive percentile 95% interval on mean game difference. Opportunities within a rally are dependent. Height is a post-play native receive-line diagnostic at y=-1.10 m.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("sources", type=Path, nargs="+")
    args = p.parse_args()
    report(args.sources, args.out)
