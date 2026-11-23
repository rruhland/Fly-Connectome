"""Grow a continuous camera-to-joint memory outward from actual legal returns."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt25_pose_demonstrations import episode
from .tt26_grip_course import ReferenceReceiver, load_match
from .tt27_flight_feedback import OUT, SOURCE, load_bank


def bank():
    with np.load(SOURCE / "match-broader.npz", allow_pickle=False) as f:
        return tuple(f[k].copy() for k in ("keys", "goals", "scale"))


def broaden(shot, rng, level):
    changed = shot.copy()
    for prefix, sigma in (
        ("pos", [0.07, 0.08, 0.027]),
        ("vel", [0.26, 0.25, 0.18]),
        ("w_vel", [6.0, 8.0, 9.0]),
    ):
        for d, a in zip(rng.normal(0, np.array(sigma) * level), "xyz", strict=True):
            changed[prefix + "_" + a] += float(d)
    return changed


def pick(key, keys, scale):
    return int(np.argmin(np.linalg.norm((keys - key) / scale, axis=1)))


def teach(args):
    path, start = args
    shots, _ = load_match(path)
    ids = [r["id"] for r in load_bank()["rows"]]
    keys, goals, scale = bank()
    nominal = shots[ids[start]]
    rng = np.random.default_rng(981605 + start)
    arena = ReferenceReceiver()
    rows = []
    parents = [nominal]
    try:
        for t in range(18):
            # Keep expanding from successful transitions; retain the original as a restart.
            parent = parents[int(rng.integers(len(parents)))]
            shot = broaden(parent, rng, (0.60, 1.0, 1.40)[t % 3])
            _, key = episode(arena, shot, collect=True)
            if key is None:
                rows.append(dict(shot=shot, attempts=[], key=None, success=False))
                continue
            nearest = pick(key, keys, scale)
            choices = [goals[nearest], goals[start]]
            width = np.array([0.19, 0.13, 0.19, 0.21, 0.22])
            for j in range(9):
                center = goals[nearest] if j % 3 else goals[start]
                choices.append(center + rng.normal(0, width * (0.6 if j % 2 == 0 else 1.15)))
            attempts = []
            successful = False
            for goal in choices:
                if attempts and np.array_equal(attempts[-1]["goal"], goal):
                    continue
                result, _ = episode(arena, shot, goal=goal)
                attempts.append(
                    dict(
                        goal=goal.tolist(),
                        legal=result["legal"],
                        contact=result["contact"],
                        frames=result["frames"],
                        referee=result["events"],
                    )
                )
                if result["legal"]:
                    successful = True
                    keys = np.vstack((keys, key))
                    goals = np.vstack((goals, goal))
                    parents.append(shot)
                    break
            rows.append(dict(shot=shot, attempts=attempts, key=key.tolist(), success=successful))
    finally:
        arena.close()
    print(
        "practice", start, ids[start], len(parents) - 1, sum(len(r["attempts"]) for r in rows), flush=True
    )
    return dict(parent=ids[start], rows=rows)


def strategy(key, keys, goals, scale, original, name):
    distances = np.linalg.norm((keys - key) / scale, axis=1)
    if name == "original":
        return goals[np.argmin(distances[:original])]
    order = np.argsort(distances)
    if name == "expanded":
        return goals[order[0]]
    # Local averaging is restricted to motor neighbors so opposing motions cannot cancel out.
    compatible = [j for j in order[:9] if np.linalg.norm(goals[j] - goals[order[0]]) < 0.55]
    weights = 1.0 / (0.035 + distances[compatible]) ** 2
    return np.average(goals[compatible], axis=0, weights=weights)


def run(path, workers):
    keys, goals, scale = bank()
    original = len(keys)
    cache = OUT / "growing-practice.json.gz"
    if not cache.exists():
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            training = list(pool.map(teach, [(path, n) for n in range(original)]))
        _write(cache, dict(training=training))
    source = _read(cache)["training"]
    for group in source:
        for r in group["rows"]:
            if r["success"]:
                keys = np.vstack((keys, r["key"]))
                goals = np.vstack((goals, r["attempts"][-1]["goal"]))
    np.savez_compressed(
        OUT / "growing-receiver.npz", keys=keys, goals=goals, scale=scale, original=original
    )
    shots, _ = load_match(path)
    roster = _read(SOURCE / "fresh-recordings.json.gz")["records"][0]["rows"]
    donors = [r["id"] for r in load_bank()["rows"]]
    rng = np.random.default_rng(147931)
    held = [(r["group"], shots[r["id"]]) for r in roster]
    held += [("wide_variation", broaden(shots[donors[i % len(donors)]], rng, 1.5)) for i in range(76)]
    arena = ReferenceReceiver()
    rows = []
    try:
        for group, shot in held:
            blank, k = episode(arena, shot, collect=True)
            scores = {}
            same = {}
            for name in ("original", "expanded", "blend"):
                if k is None:
                    scores[name] = blank
                    continue
                target = strategy(k, keys, goals, scale, original, name)
                identity = tuple(target.tolist())
                if identity not in same:
                    scored, _ = episode(arena, shot, goal=target)
                    if scored["legal"]:
                        replay, _ = episode(
                            arena,
                            shot,
                            choose_goal=lambda v: strategy(v, keys, goals, scale, original, name),
                        )
                        if scored != replay:
                            raise ValueError("Live image controller failed to reproduce this return")
                    same[identity] = scored
                scores[name] = same[identity]
            rows.append(dict(group=group, id=shot["id"], results=scores))
    finally:
        arena.close()
    _write(OUT / "growing-held.json.gz", dict(rows=rows, n_examples=len(keys)))
    for group in ("nearby", "unrestricted", "wide_variation"):
        print(
            group,
            {p: sum(r["results"][p]["legal"] for r in rows if r["group"] == group) for p in scores},
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    options = parser.parse_args()
    run(options.feeds, options.workers)
