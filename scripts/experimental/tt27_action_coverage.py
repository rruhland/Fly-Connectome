"""Bound what the learned motor repertoire can return and test four-mistake exploration."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt25_pose_demonstrations import episode
from .tt26_grip_course import ReferenceReceiver, load_match
from .tt27_flight_feedback import OUT
from .tt27_outcome_receiver import independent


def trial(args):
    path, shard = args
    lookup, _ = load_match(path)
    used = {r["id"] for r in _read(OUT / "growing-confirmation.json.gz")["rows"]}
    for block in _read(OUT / "lateral-confirmation.json.gz")["blocks"]:
        used.update(r["id"] for r in block["rows"])
    shots = independent(lookup, seed=744817, excluded=used, near_count=1, unrestricted_count=19)
    with np.load(OUT / "lateral-memory.npz") as f:
        keys, goals, scale = (f[n] for n in ("keys", "goals", "scale"))
    distances = np.linalg.norm(goals[:, None] - goals[None, :], axis=2)
    arena = ReferenceReceiver()
    rows = []
    try:
        for group, s in shots[shard::3]:
            blank, key = episode(arena, s, collect=True)
            results = [episode(arena, s, goal=a)[0] if key is not None else blank for a in goals]
            optical = (
                np.linalg.norm((keys - key) / scale, axis=1) if key is not None else np.zeros(len(goals))
            )
            costs = optical.copy()
            observed = []
            # The live branch observes only its own previous trial. A legal trial ends exploration.
            for _ in range(5):
                a = int(np.argmin(costs))
                score = results[a]
                replay, _ = episode(arena, s, choose_goal=lambda k: goals[a])
                if replay != score:
                    raise ValueError("Failed live replay")
                observed.append(dict(action=a, legal=score["legal"], contact=score["contact"]))
                if score["legal"]:
                    break
                # Try a nearby configuration after physical contact. Spread exploration after a clean miss.
                if score["contact"]:
                    costs += np.minimum(distances[a], 1.8) * 0.55
                else:
                    costs += np.exp(-(distances[a] ** 2) / (2 * 0.5**2)) * 1.2
                costs[a] = np.inf
            rows.append(dict(id=s["id"], group=group, scores=results, online=observed))
    finally:
        arena.close()
    return rows


def run(path):
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as pool:
        rows = sum(pool.map(trial, [(path, j) for j in range(3)]), [])
    _write(OUT / "action-coverage.json.gz", dict(seed=744817, rows=rows))
    for group in ("nearby", "unrestricted"):
        group_rows = [r for r in rows if r["group"] == group]
        print(
            group,
            len(group_rows),
            sum(r["online"][0]["legal"] for r in group_rows),
            sum(r["online"][-1]["legal"] for r in group_rows),
            sum(any(s["legal"] for s in r["scores"]) for r in group_rows),
            sum(any(s["contact"] for s in r["scores"]) for r in group_rows),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    run(parser.parse_args().feeds)
