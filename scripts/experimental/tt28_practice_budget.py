"""Allocate 192 practice episodes between bouncing and online regulation receiving."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt25_pose_demonstrations import episode
from .tt26_grip_course import Grip, ReferenceReceiver, load_match
from .tt26_receiving_speed import draw
from .tt27_flight_feedback import load_bank
from .tt27_outcome_receiver import independent
from .tt28_dose import MOUNTS, OUTPUT, target_banks
from .tt28_dose_online import DISPLACEMENTS, Outcomes

BUDGET = 192


def schedule(lookup, ids):
    rng = np.random.default_rng(280388641)
    train = independent(lookup, seed=2814320, near_count=1, unrestricted_count=19)
    new = independent(
        lookup, seed=2818932, excluded=[s["id"] for _, s in train], near_count=2, unrestricted_count=38
    )
    rows = []
    for tick in range(BUDGET):
        choices = ids if tick % 2 == 0 else [s["id"] for _, s in train]
        rows.append(("training", draw(lookup, choices, rng, tick)))
    rows += [("practice_variation", draw(lookup, ids, rng, t)) for t in range(48)]
    return rows + new


def run(path, tool, long_only):
    lookup, (_, _, scale) = load_match(path)
    library = load_bank()["rows"]
    keys, goals = [np.stack([r[n] for r in library]) for n in ("key", "goal")]
    cases = schedule(lookup, [r["id"] for r in library])
    doses = [192] if long_only else [0, 16, 64]
    source = _read(OUTPUT / f"dose-solo-{tool}.json.gz")["checkpoints"]
    arena = ReferenceReceiver()
    Grip(arena.model).set(*MOUNTS[tool])
    models = {d: Outcomes() for d in doses}
    banks = target_banks(arena, goals, {d: source[str(d)] for d in doses})
    receipt = []
    try:
        for t, (group, s) in enumerate(cases):
            active = [d for d in doses if t >= d]
            if not active:
                continue
            blank, k = episode(arena, s, collect=True)
            shared = {}
            arms = {}
            for d in active:
                model = models[d]
                local = t - d
                act = None
                score = blank
                if k is not None:
                    q = k / scale
                    # Fixed first sweep followed by sparse exploration. Regulation outcomes
                    # are observed only from the chosen action, with no access to contact labels.
                    rng = np.random.default_rng(2804531 + t)
                    if t < BUDGET and local < len(DISPLACEMENTS):
                        act = local
                    elif t < BUDGET and rng.random() < 0.28:
                        act = int(rng.integers(len(DISPLACEMENTS)))
                    else:
                        act = model.choose(q)

                    def pick(image):
                        return (
                            banks[d][np.argmin(np.linalg.norm((keys - image) / scale, axis=1))]
                            + DISPLACEMENTS[act]
                        )

                    goal = pick(k)
                    if tuple(goal) not in shared:
                        score = episode(arena, s, goal=goal)[0]
                        if score["legal"] and score != episode(arena, s, choose_goal=pick)[0]:
                            raise ValueError("Camera impulse failed replay")
                        shared[tuple(goal)] = score
                    score = shared[tuple(goal)]
                    if t < BUDGET:
                        model.update(q, act, score["legal"])
                arms[str(d)] = dict(
                    action=act, legal=score["legal"], contact=score["contact"], events=score["events"]
                )
            receipt.append(dict(t=t, group=group, shot=s, results=arms))
        _write(
            OUTPUT / f"budget-{tool}-{'long' if long_only else 'mixed'}.json.gz",
            dict(budget=BUDGET, doses=doses, rows=receipt),
        )
    finally:
        arena.close()
    for g in ("practice_variation", "nearby", "unrestricted"):
        print(
            tool,
            g,
            {d: sum(r["results"][str(d)]["legal"] for r in receipt if r["group"] == g) for d in doses},
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("tool", choices=MOUNTS)
    p.add_argument("--long-only", action="store_true")
    a = p.parse_args()
    run(a.feeds, a.tool, a.long_only)
