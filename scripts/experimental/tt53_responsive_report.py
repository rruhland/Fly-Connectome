"""Summarize equal-exposure practice and matched full-game returns on held opening recordings."""

import argparse
from collections import Counter

import numpy as np

from . import tt53_responsive_school as school
from .tt4_transfer_trial import _hash, _read, _write


def scores(path):
    doc = _read(path)
    n = len(doc["rows"])
    return (
        doc["cards"][:n],
        np.array([r["score"]["learner"] for r in doc["rows"]]),
        sum(r["runtime_cpu_s"] for r in doc["rows"]),
    )


def summarize(v):
    return dict(
        games=len(v),
        returns=int(sum(v)),
        median=float(np.median(v)),
        best=int(max(v)),
        zeros=int(sum(v == 0)),
        at_least_five=int(sum(v >= 5)),
        at_least_fifteen=int(sum(v >= 15)),
    )


def same_arrivals(left, right):
    before, after = _read(left), _read(right)
    counts = Counter()
    different_commands = Counter()
    for u, v in zip(before["rows"], after["rows"], strict=True):
        for a, b in zip(u["placements"]["delivered"], v["placements"]["delivered"]):
            if not a.get("near_bounce") or not b.get("near_bounce"):
                continue
            if (
                a["far_contact"] != b["far_contact"]
                or a["near_bounce"] != b["near_bounce"]
            ):
                continue
            result = (
                ("landed" if a["near_landed"] else "missed")
                + "_to_"
                + ("landed" if b["near_landed"] else "missed")
            )
            counts[result] += 1
            start = a["far_contact"]["step"] - 1
            # Include commands between the far hit and the first arm hit, or
            # the end of the shorter tape when neither arm touched the ball.
            end = min(
                (o["near_contact"]["step"] if o["near_contact"] else len(r["actions"]))
                for o, r in ((a, u), (b, v))
            )
            if u["actions"][start:end] != v["actions"][start:end]:
                different_commands[result] += 1
    return dict(all=dict(counts), changed_receiving_actions=dict(different_commands))


def compare(paths, size):
    groups = {}
    manifest = {}
    reference = None
    for name, path in paths.items():
        cards, v, cpu = scores(path)
        if len(cards) != size or (reference is not None and reference != cards):
            raise ValueError("Assessment incomplete or stream mismatch")
        reference = cards
        groups[name] = v
        manifest[name] = dict(
            file=path.name, checksum=_hash(path), **summarize(v), cpu_seconds=cpu
        )
    ids = np.array([c["id"] for c in reference])
    distinct = np.unique(ids)
    comparisons = {}
    for name in groups:
        if name == "responsive":
            continue
        delta = groups["responsive"] - groups[name]
        by_release = np.array([delta[ids == i].mean() for i in distinct])
        rng = np.random.default_rng(530371)
        boot = rng.choice(by_release, (10000, len(by_release))).mean(axis=1)
        comparisons[name] = dict(
            better=int(sum(delta > 0)),
            worse=int(sum(delta < 0)),
            ties=int(sum(delta == 0)),
            same_physical_arrivals=same_arrivals(paths[name], paths["responsive"]),
            mean_difference=float(np.mean(delta)),
            release_cluster_bootstrap_interval=np.quantile(
                boot, [0.025, 0.975]
            ).tolist(),
        )
    return dict(cohort=len(distinct), groups=manifest, comparisons=comparisons)


def run(transfer):
    folder = school.ROOT
    runs = {}
    for name in ("ordinary", "responsive"):
        p = folder / f"{name}-live.json.gz"
        d = _read(p)
        _, v, cpu = scores(p)
        runs[name] = dict(
            **summarize(v),
            budget=d["exposure"]["budget"],
            offers=d["exposure"]["total"],
            checksum=_hash(p),
            cpu_seconds=cpu,
        )
    if transfer:
        result = compare({s: folder / f"{s}-transfer.json.gz" for s in runs}, 64)
        _write(folder / "transfer-results.json", result)
    else:
        paths = {s: folder / f"{s}-frozen.json.gz" for s in runs}
        result = compare(
            dict(before=school.season.ROOT / "before.json.gz", **paths), 24
        )
        _write(folder / "assessment.json", dict(practice=runs, assessment=result))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transfer", action="store_true")
    run(p.parse_args().transfer)
