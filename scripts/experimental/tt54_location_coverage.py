"""Map actual incoming trajectories to physical touches, departures and legal returns."""

from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt54_earned_low")


def region(c):
    if c is None:
        return "missing", "missing"
    x, y, z = c["xyz"]
    vx, vy, vz = c["velocity"]
    t = (-1.53 - y) / (0.98 * vy)
    x, z = x + vx * t, z + vz * t - 4.905 * t * t
    return (
        "center" if abs(x) < 0.10 else "wing",
        "low" if z < 1.00 else "mid" if z < 1.05 else "high",
    )


def audit(p):
    d = _read(p)
    groups = defaultdict(Counter)
    for r in d["rows"]:
        separation = [
            e
            for e in r["events"]
            if e["kind"] == "separation" and e.get("collider") == "learner_racket"
        ]
        times = [e["time"] for e in separation]
        last = r["events"][-1]
        for o in r["placements"]["delivered"]:
            if not o.get("near_bounce"):
                continue
            a = groups["/".join(region(o.get("receive_line")))]
            a["offered"] += 1
            if o["near_contact"]:
                a["touched"] += 1
                i = bisect_left(times, o["near_contact"]["time"])
                if i < len(times) and times[i] - o["near_contact"]["time"] < 0.09:
                    v = separation[i]["velocity"]
                    a["forward_after_contact"] += int(v[1] > 0.5)
                    a["forward_over_2mps"] += int(v[1] > 2)
            if o["near_landed"]:
                a["legal"] += 1
            elif last["kind"] == "point":
                a["fault_" + last["reason"]] += 1
    return dict(source=str(p), sha256=_hash(p), groups=groups)


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True, parents=True)
    out = dict(
        definition="Evaluator extrapolation to y=-1.53 from measured position and velocity at y=-1.10. Center is |x|<0.10 m. Low is z<1.00, mid [1.00,1.05), high >=1.05. Missing crossings remain in totals.",
        samples=[
            audit(season.ROOT / "learning.json.gz"),
            audit(Path("artifacts/tt48_transfer/continued-64.json.gz")),
        ],
    )
    _write(ROOT / "location-coverage.json", out)
