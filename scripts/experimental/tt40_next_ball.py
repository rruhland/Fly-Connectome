"""Can an arm pick a playable return from its own outgoing shot before the next random draw?"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from .tt4_transfer_trial import _hash, _read, _write
from .tt39_skilled_memory import NAMES, ROOT
from .tt39_opportunity_map import projected_height


def collect(exclude):
    seen = {}
    sources = {}
    for name in NAMES:
        path = ROOT / (name + ".json.gz")
        if not path.exists():
            continue
        data = _read(path)
        if len(data["rows"]) != len(data["cards"]):
            continue
        sources[name] = _hash(path)
        for card, row in zip(data["cards"], data["rows"], strict=True):
            if card["id"] in exclude or row.get("direct_fallback_ticks", 0):
                continue
            delivered = row["placements"]["delivered"]
            moves = row.get("records", [])
            changed = [
                s["tick"] for s in row.get("servo", []) if s["status"] == "corrected"
            ]
            actual = {
                round(s["hit"] * 480): s
                for s in row["score"]["shots"]
                if s["side"] == 1
            }
            for last, new in zip(delivered, delivered[1:]):
                line = new.get("receive_line")
                if (
                    not last.get("near_landed")
                    or not new.get("near_bounce")
                    or not line
                ):
                    continue
                contact = last["near_contact"]
                tick = contact["time"] * 60
                candidates = [m for m in moves if 0 < tick - m["tick"] < 25]
                if not candidates:
                    continue
                move = candidates[-1]
                if (
                    move["score"] is None
                    or any(move["tick"] <= t < tick for t in changed)
                    or move.get("impact_margin")
                    or move.get("camera_rescue")
                    or (move.get("late_repair") or {}).get("status") == "corrected"
                ):
                    continue
                shot = actual[round(contact["time"] * 480)]
                sig = np.r_[
                    shot["contact_xyz"],
                    shot["landing_xyz"],
                    shot["outgoing_mps"],
                    shot["spin_rad_s"],
                ]
                key = (
                    card["id"],
                    card["seed"],
                    card["reply_seed"],
                    last["leg"],
                    *np.round(sig, 7),
                )
                seen.setdefault(
                    key,
                    (
                        card["id"],
                        move["score"][3:13],
                        sig.tolist(),
                        [projected_height(line), float(new["near_landed"])],
                    ),
                )
    groups, x, actual, y = map(np.asarray, zip(*seen.values(), strict=True))
    return groups, x, actual, y, sources


def estimate(x, y, query):
    # Ridge and a bounded local residual permit smooth nonlinear response without
    # fitting a dense interaction model to sparse outcomes.
    mu = np.mean(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), 0.001)
    z, q = (x - mu) / scale, (query - mu) / scale
    design = np.c_[np.ones(len(z)), z]
    b = np.linalg.solve(
        design.T @ design + np.diag(np.r_[0.01, np.full(z.shape[1], 45.0)]),
        design.T @ y,
    )
    residual = y - design @ b
    weights = np.exp(-0.65 * cdist(q, z, "sqeuclidean") / z.shape[1])
    return np.c_[np.ones(len(q)), q] @ b + weights @ residual / (
        10 + weights.sum(axis=1, keepdims=True)
    )


def run(output, plan):
    exclude = {c["id"] for c in _read(plan)["assessment"]}
    group, optical, actual, y, sources = collect(exclude)
    blocks = np.array_split(
        np.random.default_rng(400127).permutation(np.unique(group)), 6
    )
    scores = {}
    for name, x in [
        ("constant", None),
        ("candidate_only", optical),
        ("actual_outbound_diagnostic", actual),
    ]:
        p = np.empty_like(y)
        for block in blocks:
            test = np.isin(group, block)
            p[test] = (
                estimate(x[~test], y[~test], x[test])
                if x is not None
                else y[~test].mean(axis=0)
            )
        order = np.argsort(p[:, 0])
        quantiles = np.array_split(order, 4)
        scores[name] = dict(
            height_rmse_m=float(np.mean((p[:, 0] - y[:, 0]) ** 2) ** 0.5),
            next_legal_brier=float(np.mean((p[:, 1].clip(0, 1) - y[:, 1]) ** 2)),
            low_frequency_in_predicted_height_quartiles=[
                float(np.mean(y[ix, 0] < 0.99)) for ix in quantiles
            ],
        )
    _write(
        output,
        dict(
            pairs=len(y),
            unique_releases=len(set(group)),
            rate_next_legal=float(y[:, 1].mean()),
            low_rate=float(np.mean(y[:, 0] < 0.99)),
            sources=sources,
            held_plan=_hash(plan),
            scores=scores,
            selection="Distinct issued and legally answered physical arm returns with a measured subsequent arrival. Split by original release. Actual outgoing shot features and receiving height are post-hoc diagnostics.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("plan", type=Path)
    a = p.parse_args()
    run(a.output, a.plan)
