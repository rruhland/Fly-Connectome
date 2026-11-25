"""Validate and summarize same-draw experimental replies and achieved envelope coverage."""
import argparse
from pathlib import Path

import numpy as np

from . import tt45_tactical_coverage as study
from . import tt45_tactical_goal as tactic
from .tt4_transfer_trial import _hash, _read, _write


def matrix(plan, measured, goal, output):
    method, data = _read(plan), _read(measured)
    if data["plan_sha256"] != _hash(plan):
        raise ValueError("Native replies were generated under another intervention schedule")
    joined = []
    by_case = []
    by_region = dict(inside=[], outside=[])
    envelope = tactic.Envelope(_read(goal))
    for practice, trial in zip(method["cases"], data["results"]):
        if practice["id"] != trial["id"]:
            raise ValueError("Native reply roster differs")
        a, b = [trial["worlds"][n]["trials"][1:] for n in ("low_risk", "high_risk")]
        regs = [envelope.contains(np.array(practice[n]["outgoing"])) for n in ("low_risk", "high_risk")]
        for p, q in zip(a, b, strict=True):
            if p["target"] != q["target"]:
                raise ValueError("Different random draws in physical comparison")
            joined.append((p, q))
            for row, reg in zip((p, q), regs, strict=True):
                by_region["inside" if reg else "outside"].append(row)
        good = [i for i in range(len(a)) if a[i]["height"] is not None and b[i]["height"] is not None]
        gaps = [a[i]["height"] - b[i]["height"] for i in good]
        by_case.append(dict(id=practice["id"], hard=practice["hard"],
                            in_learned_envelope=regs, lower_risk=summary(a), higher_risk=summary(b),
                            height_gain_m=float(np.mean(gaps)) if gaps else None,
                            min_gain_m=min(gaps, default=None), max_gain_m=max(gaps, default=None)))
    if not joined:
        raise ValueError("No physical counterfactuals yet")
    x = np.array([p["target"] for p, _ in joined])
    low, high = map(list, zip(*joined, strict=True))
    complete = dict(complete=data["complete"], physical_draw_pairs=len(joined), releases=len(by_case),
                    min_request=x.min(axis=0).tolist(), max_request=x.max(axis=0).tolist(),
                    distinct_requested=len(np.unique(x, axis=0)), lower_risk=summary(low),
                    higher_risk=summary(high), region={k: summary(v) for k,v in by_region.items()},
                    by_release=by_case, source_sha256=_hash(measured), goal_sha256=_hash(goal))
    _write(output, complete)
    print(complete, flush=True)


def summary(rows):
    z = [r["height"] for r in rows if r["height"] is not None]
    return dict(n=len(rows), answered=sum(r["answered"] for r in rows), tracked=len(z),
                low=sum(h < study.THRESHOLD for h in z),
                height_mean_m=float(np.mean(z)) if z else None,
                min_m=min(z, default=None))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=study.ROOT / "physical-arc-plan.json.gz")
    p.add_argument("--replies", type=Path, default=study.ROOT / "physical-replies.json.gz")
    p.add_argument("--goal", type=Path, default=study.ROOT / "learned-envelope.json")
    p.add_argument("--output", type=Path, default=study.ROOT / "physical-reply-report.json")
    args = p.parse_args()
    matrix(args.plan, args.replies, args.goal, args.output)
