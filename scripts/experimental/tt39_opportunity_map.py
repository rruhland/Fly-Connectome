"""Measure which incoming balls end continuous points, using evaluator-only physical telemetry."""

import argparse
from bisect import bisect_right
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit


EDGES = (0.99, 1.04, 1.09)
BANDS = ("below_0.99", "0.99_to_1.04", "1.04_to_1.09", "at_least_1.09")


def projected_height(crossing):
    _, y, z = crossing["xyz"]
    _, vy, vz = crossing["velocity"]
    if vy >= 0 or y < -1.53:
        raise ValueError("Expected an inbound trajectory crossing the receiving line")
    dt = (-1.53 - y) / (0.98 * vy)
    return z + dt * vz - 4.905 * dt**2


def measure(source):
    d = _read(source)
    if len(d["rows"]) != len(d["cards"]):
        raise ValueError(f"Incomplete course {source}")
    totals = {
        k: dict(offered=0, touched=0, legal=0, answered=0)
        for k in (*BANDS, "unobserved")
    }
    for c, r in zip(d["cards"], d["rows"], strict=True):
        placements = r["placements"]
        if c.get("opening", "random") != "random" or any(
            v["draw"] != "random" for v in placements["requested"]
        ):
            raise ValueError("Expected continuously randomized play")
        deliveries = placements["delivered"]
        score = audit(r["events"], opening_side=1)
        if sum(bool(v.get("near_bounce")) for v in deliveries) != score["opponent"]:
            raise ValueError("Opportunities disagree with the physical audit")
        if sum(bool(v.get("near_landed")) for v in deliveries) != score["learner"]:
            raise ValueError("Returns disagree with the physical audit")
        for i, v in enumerate(deliveries):
            if not v.get("near_bounce"):
                continue
            crossing = v.get("receive_line")
            name = (
                BANDS[bisect_right(EDGES, projected_height(crossing))]
                if crossing
                else "unobserved"
            )
            counts = totals[name]
            counts["offered"] += 1
            counts["touched"] += int(bool(v.get("near_contact")))
            counts["legal"] += int(bool(v.get("near_landed")))
            counts["answered"] += int(
                bool(v.get("near_landed"))
                and i + 1 < len(deliveries)
                and bool(deliveries[i + 1].get("near_bounce"))
            )
    return dict(
        file=str(source), sha256=_hash(source), points=len(d["rows"]), bands=totals
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("sources", nargs="+", type=Path)
    a = p.parse_args()
    _write(
        a.out,
        {
            "method": "Offline height projection to y=-1.53 m from realized post-bounce flight at y=-1.10 m. Counts require a legal opponent delivery. These estimates are unavailable to the learner.",
            "courses": [measure(f) for f in a.sources],
        },
    )
