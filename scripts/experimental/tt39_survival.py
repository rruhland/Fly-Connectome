"""Summarize whole-physical-point survival, opportunity conversion and matched learning gains."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit


def summary(path):
    d = _read(path)
    if len(d["cards"]) != len(d["rows"]):
        raise ValueError(f"Incomplete course {path}")
    counts = []
    opportunities = []
    choices = 0
    wall = 0.0
    simulated = 0.0
    starts = []
    endings = {}
    for c, r in zip(d["cards"], d["rows"], strict=True):
        if c.get("opening", "random") != "random" or c.get("profile", "reach") != "reach":
            raise ValueError("Random first and all later deliveries required")
        source = audit(r["events"], opening_side=1)
        if (source["learner"], source["opponent"]) != (
            r["score"]["learner"],
            r["score"]["opponent"],
        ):
            raise ValueError("Referee score mismatch")
        if any(p["draw"] != "random" for p in r["placements"]["requested"]):
            raise ValueError("Non-random opponent request")
        counts.append(source["learner"])
        opportunities.append(source["opponent"])
        choices += sum(m["choices"] for m in r.get("records", [])) or r.get(
            "planner", {}
        ).get("native_candidates", 0)
        wall += r.get("cpu_s", r.get("runtime_cpu_s", 0))
        simulated += r.get("simulation_s", r["events"][-1]["time"])
        starts.append(c["id"])
        reason = r["events"][-1].get("reason", "capped")
        endings[reason] = endings.get(reason, 0) + 1
    n = len(counts)
    a = np.array(counts)
    opp = np.array(opportunities)
    return {
        "source": path.name,
        "sha256": _hash(path),
        "n": n,
        "own": sum(counts),
        "opportunities": sum(opportunities),
        "first": int(sum(a > 0)),
        "later": int(np.maximum(a - 1, 0).sum()),
        "longest_own": int(max(a)),
        "probability_by_length": {
            str(k): float(np.mean(a >= k)) for k in [1, 2, 5, 10, 15, 20, 25, 50, 100]
        },
        "returned_when_opponent_returned": float(a.sum() / max(1, opp.sum())),
        "own_by_point": counts,
        "opportunities_by_point": opportunities,
        "opening": starts,
        "terminal_reasons": endings,
        "native_choices": choices,
        "cpu_seconds": wall,
        "simulated_seconds": simulated,
    }


def compare(paths, out):
    docs = [_read(p) for p in paths]
    if any(d["cards"] != docs[0]["cards"] for d in docs[1:]):
        raise ValueError("Mismatched initial assignments")
    results = [summary(p) for p in paths]
    a = np.asarray(results[0]["own_by_point"])
    rng = np.random.default_rng(394071)
    ix = rng.integers(0, len(a), (6000, len(a)))
    for r in results[1:]:
        delta = np.array(r["own_by_point"]) - a
        r["paired_with_first"] = {
            "won": int(sum(delta > 0)),
            "tied": int(sum(delta == 0)),
            "lost": int(sum(delta < 0)),
            "mean_extra_own_per_start": float(delta.mean()),
            "point_bootstrap_mean90": np.quantile(
                delta[ix].mean(axis=1), [0.05, 0.95]
            ).tolist(),
        }
    _write(out, {"courses": results})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("courses", type=Path, nargs="+")
    a = p.parse_args()
    compare(a.courses, a.output)
