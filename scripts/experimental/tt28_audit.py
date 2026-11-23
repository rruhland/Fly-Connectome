"""Pair source checkpoints by physical launch and aggregate the TT28 evaluations."""

from collections import defaultdict

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt28_dose import DOSES, MOUNTS, OUTPUT


def paired_target():
    scores = defaultdict(list)
    for tool in MOUNTS:
        target = _read(OUTPUT / f"dose-target-{tool}.json.gz")["rows"]
        switch = _read(OUTPUT / f"switch-{tool}.json.gz")["results"]
        for a, b in zip(target, switch, strict=True):
            if a["shot"] != b["shot"] or a["group"] != b["group"]:
                raise ValueError("Gated target deck changed")
            scores[a["group"]].append(
                (a["shot"]["id"], [a["results"][str(d)]["legal"] for d in DOSES] + [b["result"]["legal"]])
            )
    output = {}
    for group, rows in scores.items():
        ids = np.array([r[0] for r in rows])
        counts = np.array([r[1] for r in rows], dtype=int)
        clusters = np.unique(ids)
        changes = counts[:, -2] - counts[:, 1]
        paired = [int(np.sum(changes[ids == k])) for k in clusters]
        sizes = np.array([sum(ids == k) for k in clusters])
        index = np.random.default_rng(280173).integers(len(clusters), size=(10000, len(clusters)))
        interval = 100 * np.percentile(
            np.array(paired)[index].sum(axis=1) / sizes[index].sum(axis=1), [2.5, 97.5]
        )
        output[group] = dict(
            n=len(rows),
            distinct_donors=len(clusters),
            legal_by_checkpoint=counts.sum(axis=0).tolist(),
            gains_192_vs_4=int(sum(changes > 0)),
            losses_192_vs_4=int(sum(changes < 0)),
            interval_points_192_vs_4=interval.tolist(),
        )
    return output


def budget():
    totals = defaultdict(lambda: defaultdict(int))
    for tool in MOUNTS:
        mixed = _read(OUTPUT / f"budget-{tool}-mixed.json.gz")["rows"]
        long = _read(OUTPUT / f"budget-{tool}-long.json.gz")["rows"]
        for d in [0, 16, 64]:
            if sum(str(d) in r["results"] and r["group"] == "training" for r in mixed) != 192 - d:
                raise ValueError("Unequal total practice")
        held = [r for r in mixed if r["group"] != "training"]
        for a, b in zip(held, long, strict=True):
            if a["shot"] != b["shot"]:
                raise ValueError("Budget deck changed")
            scores = {**a["results"], **b["results"]}
            for d, result in scores.items():
                totals[a["group"]][d] += result["legal"]
            totals[a["group"]]["n"] += 1
    return totals


def live():
    sums = defaultdict(
        lambda: defaultdict(lambda: dict(points=0, first=0, replied=0, second=0, near=0, best=0))
    )
    for m in MOUNTS:
        branches = {d: _read(OUTPUT / f"live-{m}-{d}.json.gz")["rows"] for d in (0, 16, 192)}
        if any([r["shot"] for r in v] != [r["shot"] for r in branches[0]] for v in branches.values()):
            raise ValueError("Mismatched live releases")
        for d, rows in branches.items():
            for row in rows:
                a = sums[row["group"]][d]
                r = row["result"]
                n, f = r["near"], r["far"]
                a["points"] += 1
                a["first"] += int(n > 0)
                a["replied"] += int(f > 0)
                a["second"] += int(n > 1)
                a["near"] += n
                a["best"] = max(a["best"], n + f)
    return sums


def toss():
    result = {}
    rosters = []
    for suffix in ("", "-confirm"):
        standard = _read(OUTPUT / f"toss-target{suffix}.json.gz")["results"]
        graduated = _read(OUTPUT / f"toss-target-staged{suffix}.json.gz")["results"]
        totals = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
        ids = {r["id"] for r in standard if r["coached"] is None}
        rosters.append(ids)
        for a, b in zip(standard, graduated, strict=True):
            if (a["id"], a["coached"]) != (b["id"], b["coached"]):
                raise ValueError("Toss cohorts differ")
            steps = [{**x, **y} for x, y in zip(a["rows"], b["rows"], strict=True)]
            for n in steps[0]:
                t = totals[a["group"]][n]
                t[0] += int(any(row[n]["legal"] for row in steps[:3]))
                t[1] += int(any(row[n]["legal"] for row in steps[:8]))
                t[2] += sum(row[n]["legal"] for row in steps[8:])
        result[suffix or "first"] = totals
    if rosters[0] & rosters[1]:
        raise ValueError("Reused original donor")
    return result


if __name__ == "__main__":
    _write(
        OUTPUT / "assessment.json",
        dict(
            checkpoint_order=[*DOSES, "switch"],
            target=paired_target(),
            budget=budget(),
            live=live(),
            toss=toss(),
        ),
    )
