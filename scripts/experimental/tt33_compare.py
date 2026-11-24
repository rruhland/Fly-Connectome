"""Verify matched opponent deliveries and aggregate rescued returns and closed-loop controls."""

import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit


def delivery_signature(leg):
    return [
        [round(v, 7) for v in [e["time"], *e["ball"], *e["velocity"]]]
        for e in (leg["far_contact"], leg["near_bounce"])
    ]


def score(path):
    source = _read(path)
    if len(source["cards"]) != len(source["rows"]):
        raise ValueError(f"Incomplete game set in {path}")
    legal = []
    first = []
    signatures = []
    for c, r in zip(source["cards"], source["rows"], strict=True):
        if (c["id"], c["seed"]) != (r["shot"]["id"], r["opponent_seed"]):
            raise ValueError("Release or seed changed")
        a = audit(r["events"], opening_side=1)
        if (a["learner"], a["returns"]) != (
            r["score"]["learner"],
            r["score"]["returns"],
        ):
            raise ValueError("Referee statistics disagree")
        legal.append(a["learner"])
        deliveries = r["placements"]["delivered"]
        if deliveries:
            item = deliveries[0]
            signatures.append(delivery_signature(item))
            first.append(int(item["near_landed"]))
        else:
            signatures.append(None)
            first.append(0)
    rows = source["rows"]
    return signatures, {
        "own": legal,
        "total_arm": sum(legal),
        "opening_success": sum(first),
        "later_arm": sum(legal) - sum(first),
        "combined": sum(r["score"]["returns"] for r in rows),
        "hash": _hash(path),
        "cpu_s": sum(r.get("process_cpu_s", r.get("runtime_cpu_s", 0)) for r in rows)
        or None,
        "simulation_s": sum(len(r["actions"]) / 60 for r in rows),
    }


def assemble(root, output):
    paths = {
        "fixed": root.parent / "tt32" / "fresh-fixed.json.gz",
        "wider": root.parent / "tt32" / "idle-fresh-wider.json.gz",
    }
    for n in (
        "joint-fresh-empty",
        "joint-fresh-raw",
        "joint-fresh-plane",
        "joint-fresh-corrected",
        "local-fresh",
        "local-post-fresh",
        "stream-fresh",
        "servo-adapt-fresh",
    ):
        paths[n] = root / (n + ".json.gz")
    reference = None
    result = {}
    for k, path in paths.items():
        signatures, result[k] = score(path)
        if reference is not None and signatures != reference:
            raise ValueError(f"First physical offers changed for {k}")
        reference = signatures
    signature, changed = score(root / "changed-direct.json.gz")
    blind = _read(root / "changed-blind.json")["rows"]
    if signature != [
        delivery_signature(r["placements"]["delivered"][0])
        if r["placements"]["delivered"]
        else None
        for r in blind
    ]:
        raise ValueError("Blind and reactive evaluation deliveries changed")
    mirror = _read(root / "reply-change.json.gz")
    _, mirror_score = score(root / "reply-change.json.gz")
    _write(
        output,
        {
            "fresh": result,
            "changed": changed,
            "changed_blind": [r["legal"] for r in blind],
            "mirror": mirror_score,
            "mirror_blind": [r["blind_arm"] for r in mirror["rows"]],
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    assemble(args.root, args.output)
