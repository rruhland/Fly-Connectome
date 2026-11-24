"""Measure whether different legal arm returns cause easier incoming flights under the same next aim."""

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt39_opportunity_map import projected_height


def files():
    keep = []
    for folder in ("tt37", "tt38", "tt39", "tt40", "tt41"):
        for p in Path("artifacts", folder).glob("*.json.gz"):
            if p.stat().st_size > 9_000_000 or not any(
                s in p.name
                for s in (
                    "joint",
                    "credit",
                    "replication",
                    "value",
                    "school",
                    "endurance",
                    "mix-rollin",
                    "preparation",
                    "direct",
                    "motor",
                    "learn",
                    "tutor",
                    "optical",
                    "servo",
                    "retained",
                    "early-",
                    "rollin",
                    "prospective",
                    "within",
                )
            ):
                continue
            keep.append(p)
    return sorted(keep)


def take(sources):
    groups = defaultdict(dict)
    use = {}
    for path in sources:
        if path.name.startswith("reply-control"):
            continue
        doc = _read(path)
        if (
            not isinstance(doc, dict)
            or not doc.get("cards")
            or len(doc.get("rows", [])) != len(doc["cards"])
        ):
            continue
        for c, r in zip(doc["cards"], doc["rows"], strict=True):
            if (
                "placements" not in r
                or "score" not in r
                or c.get("profile", "reach") != "reach"
            ):
                continue
            delivered = r["placements"].get("delivered", [])
            hits = [
                s
                for s in r.get("events", [])
                if s["kind"] == "separation" and s.get("collider") == "learner_racket"
            ]
            near_scores = [
                s for s in r["events"] if s["kind"] == "legal_return" and s["side"] == 1
            ]
            guesses = [s for s in r.get("searches", []) if s.get("target_errors")]
            for a, b in itertools.pairwise(delivered):
                if (
                    a.get("near_contact") is None
                    or not a.get("near_landed")
                    or not b.get("receive_line")
                    or b.get("leg") != a.get("leg", -1) + 1
                ):
                    continue
                t = a["near_contact"]["time"]
                e = next((s for s in hits if t <= s["time"] < t + 0.04), None)
                land = next(
                    (
                        s
                        for s in near_scores
                        if t < s["time"] < b["far_contact"]["time"]
                    ),
                    None,
                )
                if e is None or land is None:
                    continue
                last = a["far_contact"]
                key = (
                    c["id"],
                    a["leg"],
                    round(last["time"] * 480),
                    *np.round(
                        last["ball"] + last["velocity"] + last["spin"] + b["target"], 5
                    ),
                )
                optical = None
                eligible = [
                    n
                    for n in guesses
                    if t - 0.22 < n["tick"] / 60 <= t
                    and abs(t - n["tick"] / 60 - n["contact_delay_s"]) < 0.07
                ]
                if eligible:
                    optical = eligible[-1]["target_errors"]
                actual = np.r_[
                    e["ball"],
                    e["velocity"],
                    e["spin"],
                    land["ball"][:2],
                    land["time"] - e["time"],
                ]
                y = projected_height(b["receive_line"])
                arm = groups[key]
                individual = tuple(np.round(actual, 5))
                if individual not in arm or (
                    arm[individual][-1] is None and optical is not None
                ):
                    arm[individual] = (actual, y, optical)
                    use[str(path)] = True
    return groups, use


def fit(x, d):
    s = np.std(x, axis=0).clip(0.02)
    # Differencing cancels the common incoming shot, opponent body and next
    # independent aim. Swapping treatment and baseline preserves the sign.
    z = x / s
    w = np.linalg.solve(z.T @ z + 7 * np.eye(z.shape[1]), z.T @ d)
    return w / s


def validate(groups, name):
    rows = []
    for key, moves in groups.items():
        options = list(moves.values())
        if name == "candidate":
            options = [o for o in options if o[2] is not None]
        options = sorted(options, key=lambda o: tuple(o[0]))
        if len(options) < 2:
            continue
        # Give each identical incoming condition at most six comparisons and
        # keep every release on just one side of validation.
        rng = np.random.default_rng([31238, int(key[0]), int(key[1]), int(key[2])])
        pairs = [(i, j) for i in range(len(options)) for j in range(i)]
        ids = rng.choice(len(pairs), min(6, len(pairs)), replace=False)
        for k in ids:
            i, j = pairs[k]
            a, b = options[i], options[j]
            if name == "candidate":
                x, y = np.asarray(a[2]), np.asarray(b[2])
            else:
                x, y = a[0], b[0]
                if name == "translation":
                    x, y = (
                        x[[0, 1, 2, 3, 4, 5, 9, 10, 11]],
                        y[[0, 1, 2, 3, 4, 5, 9, 10, 11]],
                    )
            rows.append((key[0], x - y, a[1] - b[1]))
    if len(rows) < 10:
        return {"pairs": len(rows)}
    g, x, y = map(np.asarray, zip(*rows, strict=True))
    pred = np.zeros(len(y))
    for held in np.array_split(
        np.random.default_rng(11718).permutation(np.unique(g)), 6
    ):
        test = np.isin(g, held)
        pred[test] = x[test] @ fit(x[~test], y[~test])
    strong = np.abs(y) > 0.01
    w = fit(x, y)
    return {
        "pairs": len(y),
        "releases": len(set(g)),
        "one_cm_pairs": int(strong.sum()),
        "variation_rmse_m": float(np.sqrt(np.mean(y * y))),
        "predict_rmse_m": float(np.sqrt(np.mean((pred - y) ** 2))),
        "signed_success_above_1cm": float(
            np.mean(np.sign(pred[strong]) == np.sign(y[strong]))
        ),
        "naive_rms_prediction_m": float(np.sqrt(np.mean(pred * pred))),
        "coef": w.tolist(),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    groups, used = take(files())
    _write(
        a.out,
        {
            "opportunities": len(groups),
            "branches": sum(map(len, groups.values())),
            "matched_groups": sum(len(v) > 1 for v in groups.values()),
            "source_sha256": {p: _hash(Path(p)) for p in used},
            "scores": {
                n: validate(groups, n) for n in ("translation", "spin", "candidate")
            },
            "role": "Retrospective diagnostic across archived controllers. Actual outgoing spin and flight enter evaluation only. Data split by release.",
        },
    )
