"""Compare regulation point receipts after checking every first delivery and referee score."""

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt29_endurance import audit
from .tt31_curriculum_summary import compare_delivery, first_deliveries
from .tt34_latency import OUT


def paired(reference, candidate):
    if reference["cards"] != candidate["cards"]:
        raise ValueError("Course cards differ")
    pairs = []
    for before, after in zip(reference["rows"], candidate["rows"], strict=True):
        offers = compare_delivery(first_deliveries([before]), first_deliveries([after]))
        if not all(offers[key] == 1 for key in ("both_delivered", "target")):
            raise ValueError(f"Different physical first offer: {offers}")
        left, right = (r["placements"]["delivered"][0] for r in (before, after))
        for event in ("far_contact", "near_bounce"):
            for field, first in left[event].items():
                other = right[event][field]
                equal = (
                    first == other
                    if isinstance(first, str)
                    else np.allclose(first, other, rtol=0, atol=1e-8)
                )
                if not equal:
                    raise ValueError(f"First offer changed at {event}/{field}")
        own = []
        for row in (before, after):
            score = audit(row["events"], opening_side=1)["learner"]
            landed = sum(op["near_landed"] for op in row["placements"]["delivered"])
            if score != row["score"]["learner"] or score != landed:
                raise ValueError(
                    "Disagreement between physical returns and saved score"
                )
            own.append(score)
        pairs.append(own)
    return {
        "points": len(pairs),
        "own": [sum(x) for x in zip(*pairs, strict=True)],
        "openings": [sum(v > 0 for v in x) for x in zip(*pairs, strict=True)],
        "later": [sum(max(0, v - 1) for v in x) for x in zip(*pairs, strict=True)],
        "improved": sum(b > a for a, b in pairs),
        "tied": sum(a == b for a, b in pairs),
        "worsened": sum(b < a for a, b in pairs),
        "per_point": pairs,
    }


def read(name):
    return _read(OUT / (name + ".json.gz"))


def report():
    results = {}
    for desc, ref, candidates in [
        ("delayed_reference", "lag-frozen", ["lag-online", "lag-oracle"]),
        (
            "delayed_novel",
            "lag-novel-frozen",
            ["lag-novel-carry", "lag-novel-kinematic"],
        ),
        (
            "nominal_reference",
            "../tt33/joint-fresh-raw",
            [
                "lag-restored",
                "temporal-ordered-check",
                "temporal-shuffled-check",
                "spatial-aligned-check",
                "spatial-reverse-check",
                "spatial-visual-only-check",
                "../tt35/relative-learned-reference",
                "../tt35/relative-shuffled-reference",
            ],
        ),
        (
            "nominal_novel",
            "body-novel-stock",
            [
                "temporal-ordered-novel",
                "temporal-shuffled-novel",
                "spatial-aligned-novel",
                "spatial-visual-only-novel",
                "../tt35/relative-learned-novel",
                "../tt35/relative-shuffled-novel",
            ],
        ),
    ]:
        reference = read(ref)
        results[desc] = {c: paired(reference, read(c)) for c in candidates}
    before, after = read("switch-frozen"), read("switch-online")
    results["changing_delay"] = paired(before, after)
    blocks = []
    for i in range(0, len(after["rows"]), 8):

        def block(receipt, start=i):
            return {k: receipt[k][start : start + 8] for k in ("rows", "cards")}

        delays = after["schedule"][i : i + 8]
        if len(set(delays)) != 1 or before["schedule"][i : i + 8] != delays:
            raise ValueError("Mismatched delay courses")
        blocks.append(
            dict(delay_frames=delays[0], **paired(block(before), block(after)))
        )
    results["changing_delay_blocks"] = blocks
    _write(OUT / "zero-reward-results.json", results)


if __name__ == "__main__":
    report()
