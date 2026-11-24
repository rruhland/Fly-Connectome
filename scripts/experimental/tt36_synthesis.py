"""Reconcile component, transfer and local-learning courses from their physical action receipts."""

import argparse
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt33_joint import verify
from .tt34_results import paired
from .tt35_replies import varied_opponent
from .tt36_compare import outcomes
from .tt36_course import OUT
from .tt36_nearby import displaced


def read(name):
    return _read(OUT / f"{name}.json.gz")


def extract(doc, indices):
    return {key: [doc[key][i] for i in indices] for key in ("cards", "rows")}


def measure(ref, other):
    return {"counts": outcomes(other["rows"]), "comparison": paired(ref, other)}


def report():
    result = {}
    local = read("local-held-fixed")
    for kind in ("first", "reply"):
        indices = [i for i, x in enumerate(local["schedule"]) if x["kind"] == kind]
        ref = extract(local, indices)
        modes = [
            "local-held-fixed",
            "local-held-score",
            "local-held-dense",
            "inverse-held",
            "staged-held-score",
            "staged-held-dense",
            "staged-held-late-only",
        ]
        result[kind] = {k: measure(ref, extract(read(k), indices)) for k in modes}
    original = read("nearby-fixed")
    result["perturbed"] = {
        name: measure(original, read(f"nearby-{name}"))
        for name in ("fixed", "early", "late-only", "both")
    }
    samples = _read(OUT / "nearby-plan.json")["points"]
    distances = np.linalg.norm([s["bounce_shift_m"] for s in samples], axis=1)
    result["actual_first_bounce_shift_m"] = {
        "count": len(samples),
        "median": float(np.median(distances)),
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
    }
    for cohort in ("new", "confirm", "replies"):
        base = read(f"paired-{cohort}-fixed")
        result[f"reach_{cohort}"] = {
            m: measure(base, read(f"reach-{cohort}-{m}"))
            for m in ("aligned", "composed", "gated", "projected", "scrambled")
        }
        if cohort == "new":
            result["reach_new"]["reversed"] = measure(base, read("reach-new-reversed"))
    retained = _read(OUT.parent / "tt35/replies-nominal.json.gz")
    carry = read("reach-retained-aligned")
    if any(r["score"]["returns"] >= 16 for r in retained["rows"] + carry["rows"]):
        raise ValueError("Retention comparison reached the shorter of the two stop limits")
    result["retention"] = measure(retained, carry)
    changes = []
    phases = read("staged-practice")["cases"]
    for c in phases:
        sid = c["id"]
        source = read(f"local-practice-{sid}")
        first = source["common"] + source["forks"]["dense"]["added"]
        rows = [e["row"] for e in first + c["episodes"]]

        def first_seen(pred, rows=rows):
            return next((i + 1 for i, r in enumerate(rows) if pred(r)), None)

        changes.append(
            {
                "id": sid,
                "first_contact_attempt": first_seen(
                    lambda r: (
                        r["placements"]["delivered"][0]["near_contact"] is not None
                    )
                ),
                "first_legal_attempt": first_seen(lambda r: r["score"]["learner"] > 0),
                "practice_own": [r["score"]["learner"] for r in rows],
                "cpu_s": sum(r["runtime_cpu_s"] for r in rows),
            }
        )
    result["local_learner"] = changes
    result["gate_confirmation_usage"] = dict(
        Counter(
            r["selected_update"][0]
            for r in read("paired-confirm-gate")["rows"]
            if r["selected_update"]
        )
    )
    _write(OUT / "learning-results.json", result)


def audit():
    receipts = []
    with TemporaryDirectory() as folder:
        for name, reference in [
            ("staged-held-dense", "local-held-fixed"),
            ("nearby-both", "nearby-fixed"),
        ]:
            data = read(name)
            base = read(reference)
            changes = paired(base, data)["per_point"]
            for i, (old, new) in enumerate(changes):
                if old == new:
                    continue
                card = data["cards"][i]
                if name.startswith("nearby"):
                    p = _read(OUT / "nearby-plan.json")["points"][i]
                    variation = displaced(card, p["delta"])
                    seed = p["reply_seed"]
                else:
                    variation = nullcontext()
                    seed = data["schedule"][i]["reply_seed"]
                with variation, varied_opponent(card["index"], seed):
                    tape = Path(folder) / "tape.json.gz"
                    report_path = Path(folder) / "verify.json"
                    _write(
                        tape,
                        extract(data, [i])
                        | {"ready": data["ready"], "opening_side": 1},
                    )
                    verify(tape, report_path, cad=True)
                    r = _read(report_path)["points"][0]
                receipts.append(
                    {
                        "run": name,
                        "index": i,
                        "id": card["id"],
                        "seed": seed,
                        "before": old,
                        "after": new,
                        **r,
                    }
                )
    _write(
        OUT / "learning-audit.json",
        {
            "points": receipts,
            "source_sha256": {
                k: _hash(OUT / f"{k}.json.gz")
                for k in ("staged-held-dense", "nearby-both")
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("report", "audit"))
    args = parser.parse_args()
    (report if args.mode == "report" else audit)()
