"""Practice small continuous joint changes after a miss and test the resulting skill on fresh variations."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt25_pose_demonstrations import episode
from .tt26_grip_course import ReferenceReceiver, load_match
from .tt26_receiving_speed import draw
from .tt27_flight_feedback import OUT


def lessons():
    changes = []
    for block in _read(OUT / "growing-practice.json.gz")["training"]:
        for row in block["rows"]:
            tries = row["attempts"]
            if len(tries) < 2 or not tries[-1]["legal"]:
                continue
            for previous in tries[:-1]:
                shift = np.array(tries[-1]["goal"]) - previous["goal"]
                if 0.035 < np.linalg.norm(shift) < 0.65:
                    changes.append((row["key"], previous["goal"], bool(previous["contact"]), shift))
    return changes


def practice(row, arena, shots, library, prior):
    keys, goals, scale = library
    s = shots[row["id"]]
    _, key = episode(arena, s, collect=True)
    if key is None:
        raise ValueError("No opening observation on this shot")
    options = np.linalg.norm((keys - key) / scale, axis=1)
    modes = {}
    variations = np.random.default_rng(303130 + row["id"])
    new = [draw(shots, [row["id"]], variations, j) for j in range(4)]
    for mode in ("discrete", "prior", "axis", "reward_only"):
        costs = options.copy()
        # Contact on the physical paddle is an extra teaching signal for the first three modes.
        # The reward-only mode chooses when to vary its movement using its own attempt count.
        anchor = None
        history = []
        adjustment = 0
        for t in range(8):
            use = mode != "discrete" and anchor is not None and adjustment < 4
            if use:
                if mode == "axis":
                    d = np.zeros(5)
                    d[(adjustment // 2 + 3) % 5] = 0.17 if adjustment % 2 == 0 else -0.17
                else:
                    rank = sorted(
                        range(len(prior)),
                        key=lambda i: (
                            np.linalg.norm((np.asarray(prior[i][0]) - key) / scale)
                            + 1.9 * np.linalg.norm(np.asarray(prior[i][1]) - anchor)
                            + (0.7 if mode == "prior" and not prior[i][2] else 0.0)
                        ),
                    )
                    distinct = []
                    for i in rank:
                        d = prior[i][3]
                        if all(np.linalg.norm(d - p) > 0.08 for p in distinct):
                            distinct.append(d)
                        if len(distinct) > adjustment:
                            break
                    d = distinct[adjustment]
                goal = anchor + d
                adjustment += 1
            else:
                j = int(np.argmin(costs))
                goal = goals[j]
                costs[j] = np.inf
            score, _ = episode(arena, s, goal=goal)
            contact = score["contact"] and mode != "reward_only"
            distance = np.linalg.norm(goals - goal, axis=1)
            if contact:
                costs += np.minimum(distance, 1.8) * 0.55
            else:
                costs += 1.2 * np.exp(-(distance**2) / 0.5)
            history.append(dict(goal=goal.tolist(), legal=score["legal"], contact=score["contact"]))
            if score["legal"]:
                break
            if not use and (contact or mode == "reward_only") and mode != "discrete":
                anchor = goal.copy()
                adjustment = 0
        # Test the learner's accepted motor goal on local variations. Failed acquisitions
        # still enter the test using the first camera-selected goal, preserving the denominator.
        accepted = goal if score["legal"] else goals[np.argmin(options)]
        test = [episode(arena, v, choose_goal=lambda k: accepted)[0]["legal"] for v in new]
        modes[mode] = dict(trials=history, variations=test)
    return dict(id=row["id"], group=row["group"], modes=modes)


def shard(args):
    path, index = args
    shots, _ = load_match(path)
    rows = _read(OUT / "action-coverage.json.gz")["rows"]
    with np.load(OUT / "lateral-memory.npz") as f:
        library = tuple(f[n] for n in ("keys", "goals", "scale"))
    prior = lessons()
    arena = ReferenceReceiver()
    try:
        return [practice(r, arena, shots, library, prior) for r in rows[index::3]]
    finally:
        arena.close()


def run(path):
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as pool:
        rows = sum(pool.map(shard, [(path, j) for j in range(3)]), [])
    _write(OUT / "contact-refine.json.gz", dict(source_changes=len(lessons()), rows=rows))
    for mode in rows[0]["modes"]:
        print(
            mode,
            [
                sum(sum(t["legal"] for t in r["modes"][mode]["trials"][:n]) for r in rows)
                for n in (1, 3, 5, 8)
            ],
            sum(sum(r["modes"][mode]["variations"]) for r in rows),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    run(parser.parse_args().feeds)
