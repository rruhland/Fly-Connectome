"""Summarize conditional returns over uninterrupted games with independent starting draws."""

from collections import Counter
from pathlib import Path
import numpy as np
from .tt4_transfer_trial import _hash, _read, _write
from . import tt52_practice_dose as dose
from .tt39_opportunity_map import projected_height

ROOT = Path("artifacts/tt68_rally_survival")
SOURCES = [
    dose.season.ROOT / "learning.json.gz",
    Path("artifacts/tt48_transfer/live-exploit.json.gz"),
    dose.ROOT / "live-from-48.json.gz",
    Path("artifacts/tt53_responsive_school/ordinary-transfer.json.gz"),
    Path("artifacts/tt57_atlas_transfer/ablated.json.gz"),
]
CUTS = [0, 1, 3, 5, 10, 15, 25, 40, 60, 100, 10000]


def summarize():
    seen = set()
    bins = [Counter() for _ in CUTS[:-1]]
    nums = []
    files = {}
    for p in SOURCES:
        d = _read(p)
        n = 0
        for c, r in zip(d["cards"], d["rows"]):
            if (
                c.get("profile") != "reach"
                or c.get("opening") != "random"
                or any(z["draw"] != "random" for z in r["placements"]["requested"])
            ):
                raise ValueError("Non-broad opponent")
            key = c["id"], c["seed"], c["reply_seed"]
            if key in seen:
                continue
            seen.add(key)
            n += 1
            nums.append(r["score"]["learner"])
            successes = 0
            for o in r["placements"]["delivered"]:
                if not o.get("near_bounce"):
                    continue
                i = int(np.searchsorted(CUTS[1:], successes, side="right"))
                bucket = bins[i]
                bucket["offered"] += 1
                hit = int(bool(o["near_landed"]))
                bucket["landed"] += hit
                l = o.get("receive_line")
                if l:
                    low = projected_height(l) < 1
                    center = abs(l["xyz"][0]) < 0.10
                    cls = (
                        "central_low"
                        if low and center
                        else "wing_low"
                        if low
                        else "other"
                    )
                    bucket[cls + "_offered"] += 1
                    bucket[cls + "_landed"] += hit
                successes += hit
            if successes != r["score"]["learner"]:
                raise ValueError("Offer counting mismatch")
        files[str(p)] = dict(hash=_hash(p), independent_games=n)
    ROOT.mkdir(exist_ok=True, parents=True)
    _write(
        ROOT / "survival.json",
        dict(
            files=files,
            games=len(nums),
            legal=sum(nums),
            peak=max(nums),
            thresholds={str(n): sum(s >= n for s in nums) for n in CUTS[:-1]},
            by_prior_own_returns=[
                dict(first=a, last=b - 1, **v) for a, b, v in zip(CUTS, CUTS[1:], bins)
            ],
            scope="Only one point per unique physical release and independent opening/reply seed pair. Conditional conversion by rally age is observational and conditions on survival. Physical receive-line geometry is evaluator-only. Per-run starting motor practice can differ.",
        ),
    )


if __name__ == "__main__":
    summarize()
