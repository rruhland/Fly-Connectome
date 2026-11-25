"""Compare earned and uniformly sampled memory choices on exactly matched normal-game schedules."""

import argparse
import numpy as np

from . import tt66_memory_choice as memories
from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt68_clamp_transfer import describe
from .tt29_endurance import audit


def read(path, n):
    data = _read(path)
    rows = data["rows"][:n]
    if len(rows) != n:
        raise ValueError(f"Incomplete {path}")
    if any(audit(r["events"], opening_side=1) != r["score"] for r in rows):
        raise ValueError("Referee score mismatch")
    return data["cards"][:n], rows


def compare(paths, n):
    ref = None
    groups = {}
    totals = {}
    for k, p in paths.items():
        cards, rows = read(p, n)
        if ref is not None and ref != cards:
            raise ValueError("Mismatched random streams")
        ref = cards
        totals[k] = np.array([r["score"]["learner"] for r in rows])
        groups[k] = dict(
            **describe(rows),
            cpu_s=sum(r["runtime_cpu_s"] for r in rows),
            survival={i: int(sum(totals[k] >= i)) for i in (1, 5, 10, 15, 25, 50, 100)},
            source=str(p),
            played_prefix_sha256=digest_json(
                [
                    [c, r["actions"], r["placements"], r["events"], r["memory"]]
                    for c, r in zip(cards, rows, strict=True)
                ]
            ),
        )
    ids = np.array([c["id"] for c in ref])
    comparisons = {}
    for k in paths:
        for other in list(paths)[: list(paths).index(k)]:
            d = totals[k] - totals[other]
            by_id = [d[ids == i].mean() for i in np.unique(ids)]
            boot = (
                np.random.default_rng(669116).choice(by_id, (10000, len(by_id))).mean(1)
            )
            comparisons[f"{k}_minus_{other}"] = dict(
                total=int(d.sum()),
                better=int(sum(d > 0)),
                worse=int(sum(d < 0)),
                ties=int(sum(d == 0)),
                mean_interval=np.quantile(boot, [0.025, 0.975]).tolist(),
            )
    return dict(n=n, groups=groups, pairs=comparisons)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("n", type=int, choices=(64, 96))
    a = p.parse_args()
    paths = dict(ordinary=memories.dose.ROOT / "live-from-48.json.gz")
    if a.n == 64:
        paths["uniform"] = memories.ROOT / "uniform.json.gz"
    paths["adaptive"] = memories.ROOT / "adaptive.json.gz"
    _write(memories.ROOT / f"comparison-{a.n}.json", compare(paths, a.n))
