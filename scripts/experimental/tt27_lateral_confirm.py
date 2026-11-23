"""Confirm lateral motor practice on a second disjoint roster and physically changed conditions."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt25_pose_demonstrations import episode
from .tt26_grip_course import ReferenceReceiver, load_match
from .tt27_flight_feedback import OUT
from .tt27_growing_receiver import strategy
from .tt27_opening_course import Physics
from .tt27_outcome_receiver import independent


def roster(shots):
    old = {r["id"] for r in _read(OUT / "growing-confirmation.json.gz")["rows"]}
    return independent(shots, seed=723197, excluded=old, near_count=10, unrestricted_count=180)


def trial(args):
    path, group, changed = args
    lookup, _ = load_match(path)
    with np.load(OUT / "lateral-memory.npz") as f:
        keys, goals, scales = [f[n] for n in ("keys", "goals", "scale")]
    arena = ReferenceReceiver()
    physics = Physics(arena)
    if changed:
        physics.set(1.35, 1.04)
    cases = []
    try:
        for c, shot in roster(lookup):
            if group != c:
                continue
            blank, k = episode(arena, shot, collect=True)
            results = {}
            for name in ("original", "blend"):
                choose = lambda v: strategy(v, keys, goals, scales, 19, name)
                r = blank if k is None else episode(arena, shot, goal=choose(k))[0]
                if r["legal"] and r != episode(arena, shot, choose_goal=choose)[0]:
                    raise ValueError("Independent confirmation failed live replay")
                results[name] = r
            cases.append(dict(id=shot["id"], results=results))
        print(
            group,
            changed,
            len(cases),
            {k: sum(r["results"][k]["legal"] for r in cases) for k in results},
            flush=True,
        )
        return dict(group=group, changed=changed, rows=cases)
    finally:
        arena.close()


def run(path):
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        blocks = list(
            pool.map(
                trial, [(path, c, altered) for altered in (False, True) for c in ("nearby", "unrestricted")]
            )
        )
    _write(OUT / "lateral-confirmation.json.gz", dict(seed=723197, blocks=blocks))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    run(p.parse_args().feeds)
