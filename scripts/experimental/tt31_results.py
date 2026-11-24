"""Compare matched courses, actual left/right deliveries and independent measured receptions."""

import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit


def measured(manifest, path):
    rows = _read(path)["rows"]
    roster = [r["id"] for r in manifest["rows"]]
    if [r["shot"]["id"] for r in rows] != roster:
        raise ValueError(f"Incomplete or changed measured roster in {path}")
    scored = [audit(r["events"])["learner"] for r in rows]
    if any(v not in (0, 1) for v in scored):
        raise ValueError("Expected isolated first returns")
    left = [s["x_m"] < 0 for s in manifest["rows"]]
    return {
        "source_sha256": _hash(path),
        "actions": scored,
        "left": sum(v for v, l in zip(scored, left) if l),
        "right": sum(v for v, l in zip(scored, left) if not l),
        "total": sum(scored),
        "contacts": sum(
            sum(e["kind"] == "contact" and e["side"] == 1 for e in r["events"])
            for r in rows
        ),
        "process_cpu_s": sum(r["process_cpu_s"] for r in rows),
    }


def full(plan, path):
    d = _read(path)
    if d["cards"] != plan["cards"] or len(d["rows"]) != len(d["cards"]):
        raise ValueError(f"Incomplete or changed full-interval course in {path}")
    opportunities = {k: [0, 0] for k in ("left", "middle", "right", "unknown")}
    totals, own = [], []
    firsts, pre_contact = [], []
    later = [0, 0]
    changes = [0, 0]
    for row, card in zip(d["rows"], d["cards"], strict=True):
        if (row["shot"]["id"], row["opponent_seed"]) != (card["id"], card["seed"]):
            raise ValueError("Opponent or opening changed")
        scored = audit(row["events"], opening_side=1)
        totals.append(scored["returns"])
        own.append(scored["learner"])
        deliveries = row["placements"]["delivered"]
        first = deliveries[0]
        pre_contact.append(
            digest_json([first["target"], first["far_contact"], first["near_bounce"]])
        )
        firsts.append(int(first["near_landed"]))
        for i, leg in enumerate(deliveries):
            if i:
                later[1] += 1
                later[0] += int(leg["near_landed"])
                if {leg["observed_lane"], deliveries[i - 1]["observed_lane"]} == {
                    "left",
                    "right",
                }:
                    changes[1] += 1
                    changes[0] += int(leg["near_landed"])
            counter = opportunities[leg["observed_lane"] or "unknown"]
            counter[1] += 1
            counter[0] += int(leg["near_landed"])
    if sum(own) != sum(s for s, _ in opportunities.values()):
        raise ValueError("Placement and referee counts disagree")
    return {
        "source_sha256": _hash(path),
        "per_game": totals,
        "identical_first_offer_check": pre_contact,
        "first_ball_legal": firsts,
        "later_landed_offered": later,
        "opposite_side_landed_offered": changes,
        "retrieved": sum(r["retrieved"] for r in d["rows"]),
        "planned": sum(r["planned"] for r in d["rows"]),
        "arm_per_game": own,
        "arm": sum(own),
        "returns": sum(totals),
        "opportunities": opportunities,
        "process_cpu_s": sum(r["process_cpu_s"] for r in d["rows"]),
    }


def main(root):
    measured_plan = _read(root / "measured-openings.json")
    full_plan = _read(root / "full-plan.json.gz")
    measurements = {
        p.stem[:-5]: measured(measured_plan, p)
        for p in sorted(root.glob("measured-*.json.gz"))
    }
    courses = {
        p.stem[:-5]: full(full_plan, p)
        for p in sorted(root.glob("full-*.json.gz"))
        if "plan" not in p.name and "checkpoint" not in p.name
    }
    courses.update(
        {
            p.stem[:-5]: full(full_plan, p)
            for p in sorted(root.glob("ready-full-*.json.gz"))
        }
    )
    if any(
        item["identical_first_offer_check"]
        != courses["full-original"]["identical_first_offer_check"]
        for item in courses.values()
    ):
        raise ValueError("The physical first offer changed between learners")
    _write(root / "results.json", {"measured": measurements, "full_interval": courses})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    main(p.parse_args().directory)
