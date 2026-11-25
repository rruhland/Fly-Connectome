"""Verify paired preview policy tapes and report physical practice and transfer."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from . import tt44_preview as preview
from .tt19_view_full_game import digest_json
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_opponent import draw


def phase_report(rows, controls, phase):
    if len(rows) != len(controls):
        raise ValueError(f"Unequal {phase} releases")
    stats = [Counter(), Counter()]
    pairs = []
    for i, (old, new) in enumerate(zip(controls, rows, strict=True)):
        if old["card"] != new["card"] or old["shot"] != new["shot"]:
            raise ValueError("Different measured launch or opponent seeds")
        c = old["card"]
        p, q = (row["placements"]["requested"] for row in (old, new))
        if any(a["target"] != b["target"] for a, b in zip(p, q)):
            raise ValueError("Common opponent choices changed")
        x, y = old["actions"], new["actions"]
        t = next((j for j, (u, v) in enumerate(zip(x, y)) if u != v),
                 None if len(x) == len(y) else min(len(x), len(y)))
        if t is None and digest_json(old["events"]) != digest_json(new["events"]):
            raise ValueError("Equal command tapes changed physics")
        if t is not None and digest_json([e for e in old["events"] if e["step"] < t]) != digest_json([e for e in new["events"] if e["step"] < t]):
            raise ValueError("Physical delivery diverged before the first differing command")
        for count, row in zip(stats, (old, new), strict=True):
            u = np.asarray(row["actions"])
            if not np.isfinite(u).all() or np.abs(u).max() > .8 + 1e-12:
                raise ValueError("Out-of-range command on physical tape")
            count["joint_ticks"] += len(u)
            count["preview_ticks"] += sum(e["effect_ticks"] for e in row["episodes"])
            count["learner_returns"] += row["score"]["learner"]
            count["alternating"] += row["score"]["returns"]
            count["updates"] += row["learner_updates"]
            count["peak"] = max(count["peak"], row["score"]["learner"])
            for leg, move in enumerate(row["placements"]["requested"]):
                if move["target"] != list(draw(c, leg, "random", None)[0]):
                    raise ValueError("Independent broad random draw differs from tape")
            for e in row["episodes"]:
                delay = e["delay_ticks"]
                if phase == "train":
                    forced, arm = preview.exploration(i, e["offer"])
                    if forced != e["forced_exploration"] or (forced and preview.DELAYS[arm] != delay):
                        raise ValueError("Exploration assignment differs from frozen schedule")
                elif e["forced_exploration"] or row["learner_updates"]:
                    raise ValueError("Held games must not explore or update")
                count[f"delay_{delay}_offered"] += 1
                count["censored"] += e["earned"] is None
                if e["earned"] is not None:
                    count[f"delay_{delay}_physical_outcomes"] += 1
                    count[f"delay_{delay}_legal_hit"] += e["earned"]
        pairs.append(dict(id=c["id"], first_action_diff=t, common_random_draws=min(len(p), len(q)),
                          learner_returns=[old["score"]["learner"], new["score"]["learner"]],
                          control_event_sha256=digest_json(old["events"]),
                          treatment_event_sha256=digest_json(new["events"])))
    return {"pairs": pairs, "order": ["control", "treatment"], "totals": stats}


def report(a):
    trained, base = _read(a.treatment), _read(a.control)
    if trained["plan_sha256"] != base["plan_sha256"]:
        raise ValueError("Compared branches used different frozen plans")
    rows = {}
    for phase in ("train", "held"):
        n = min(len(trained["records"][phase]), len(base["records"][phase]))
        rows[phase] = phase_report(trained["records"][phase][:n], base["records"][phase][:n], phase)
    _write(a.out, {
        "plan_sha256": base["plan_sha256"],
        "control_sha256": _hash(a.control), "treatment_sha256": _hash(a.treatment),
        "complete": base["complete"] and trained["complete"],
        "records": rows,
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("treatment", type=Path)
    p.add_argument("control", type=Path)
    p.add_argument("out", type=Path)
    report(p.parse_args())
