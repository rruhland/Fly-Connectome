"""Retain a scored sensorimotor skill once it recurs in several independent points."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import cdist

from . import tt46_continuous_season as season
from . import tt47_season_retention as retention
from . import tt49_cumulative_memory as history
from .tt4_transfer_trial import _hash, _read, _write

course = season.course
ROOT = Path("artifacts/tt52_consolidation")
MODEL = ROOT / "recurrent.npz"


def motor(row):
    state, goals, _ = row
    n = course.mpc.OPEN_NORM
    ref = state[6:11] * n[6:11] + state[11:16] * n[11:16]
    return (
        (goals[[2, 7, 14], :5] - ref) / np.array([0.18, 0.20, 0.22, 0.25, 0.24])
    ).ravel()


def select(rows, parent):
    """Parent experience remains accessible. Additional skills need cross-point evidence."""
    extra = rows[parent:]
    x = np.array([r[0] for r in rows])
    u = np.array([motor(r) for r in rows])
    games = np.array([str((r[2][0], r[2][1])) for r in rows])
    dx = cdist(x[parent:], x) / np.sqrt(x.shape[1])
    du = cdist(u[parent:], u) / np.sqrt(u.shape[1])
    dist = np.hypot(dx, 0.65 * du)
    dist[:, :parent] = np.inf
    # A movement recurs if two other real points had a similar proprioceptive
    # situation and a similar executed motor response. Different times within
    # a single rally cannot promote each other.
    support = []
    near = []
    for j, row in enumerate(extra):
        candidate = (dx[j] < 0.40) & (du[j] < 0.60)
        candidate[:parent] = False
        groups = set(games[candidate]) - {games[parent + j]}
        support.append(len(groups))
        near.append(float(np.min(np.hypot(dx[j, :parent], 0.65 * du[j, :parent]))))
    picked = []
    priority = np.sqrt(np.minimum(6, support)) * np.minimum(1.0, near)
    for j in np.argsort(-priority, kind="stable"):
        if support[j] < 2 or near[j] < 0.28:
            continue
        if picked and np.min(dist[j, parent + np.array(picked)]) < 0.33:
            continue
        picked.append(int(j))
        if len(picked) >= 192:
            break
    return [parent + j for j in picked], support, near


def fit():
    rows, proof = history.restore_history(24)
    base = len(history.load_rows(season.CARRY))
    use, support, novel = select(rows, base)
    rng = np.random.default_rng(520382)
    matched = []
    for i in use:
        # Equal number sampled from the same acquisition window and similar
        # distance to the accepted parent's repertoire.
        pool = [j for j in range(base, len(rows)) if j not in matched]
        distance = np.array([abs(novel[j - base] - novel[i - base]) for j in pool])
        near = np.argsort(distance)[:24]
        matched.append(pool[int(rng.choice(near))])
    ROOT.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(MODEL, recurrent=use, matched=matched)
    _write(
        ROOT / "fit.json",
        dict(
            parent=base,
            new=len(rows) - base,
            selected=len(use),
            proof=proof,
            used=[
                dict(index=i, other_points=support[i - base], novelty=novel[i - base])
                for i in use
            ],
            matched=matched,
            model=_hash(MODEL),
        ),
    )


def run(a):
    rows, _ = history.restore_history(24)
    start = len(history.load_rows(season.CARRY))
    meta = _read(ROOT / "fit.json")
    with np.load(MODEL) as z:
        indices = np.asarray(z[a.selection], int)
    if meta["model"] != _hash(MODEL):
        raise ValueError("Changed selection")
    keep = [*rows[:start], *[rows[i] for i in indices]]

    def restore(actor, p, doc):
        retention.ancestral_restore(actor, p, doc, season.CARRY)
        actor.priors.online = deque(keep)

    with (
        patch.object(course, "Carried", history.Growing),
        patch.object(course, "resume", restore),
    ):
        season.run(a)
    doc = _read(a.out)
    doc.update(consolidation=_hash(MODEL), selection=a.selection)
    _write(a.out, doc)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("selection", choices=("fit", "recurrent", "matched"))
    p.add_argument("out", type=Path, nargs="?")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        carry=season.ROOT / "learning-24.json.gz",
        deck="assessment",
        mode="frozen",
        cap=240,
        resume=False,
    )
    a = p.parse_args()
    fit() if a.selection == "fit" else run(a)
