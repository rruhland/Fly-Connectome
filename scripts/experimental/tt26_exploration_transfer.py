"""Transfer which control to investigate, with zero retained correction on new grips."""

import argparse
from itertools import permutations
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt25_pose_demonstrations import episode
from .tt26_free_joints import OPTIONS
from .tt26_grip_course import Grip, OUT, PRIOR, ReferenceReceiver, load_match
from .tt26_receiving_speed import draw


def source_order(practice, rng=None):
    contrasts = []
    for record in practice:
        labels = [r["index"] for r in record["practice"]]
        rewards = [r["reward"] for r in record["practice"]]
        if rng is not None:
            rewards = rng.permutation(rewards)
        means = np.bincount(labels, weights=rewards) / np.bincount(labels)
        contrasts.append((means[1:6] - means[6:]) ** 2)
    return np.argsort(-np.mean(contrasts, axis=0), kind="stable").tolist()


def decisions(probes, order):
    seen = [0]
    decisions = [0]
    for axis in order:
        seen.extend([1 + axis, 6 + axis])
        decisions.append(max(seen, key=lambda option: probes[option]))
    return decisions


def run(path):
    shots, (keys, goals, scale) = load_match(path)
    donors = _read(PRIOR / "recorded-nearby-lessons.json.gz")["donors"]
    practice = [r for r in _read(OUT / "five-joint-practice.json.gz")["records"] if r["mount"] != 0]
    if len(practice) != 2:
        raise ValueError("Two different earlier grips are required")
    learned = source_order(practice)
    rng = np.random.default_rng(741103)
    shuffled = [source_order(practice, rng) for _ in range(128)]
    uniform = list(permutations(range(5)))
    arena = ReferenceReceiver()
    grip = Grip(arena.model)
    cases = []
    try:
        for angle in [-0.30, 0.30, -0.15, 0.15, 0.0]:
            grip.set(angle)
            rng = np.random.default_rng(470912)
            trials = []
            for i in range(56):
                # Four independently drawn eight-feed practice packets followed by 24 new test feeds.
                shot = draw(shots, donors, rng, i)
                _, key = episode(arena, shot, collect=True)
                goal = goals[np.argmin(np.linalg.norm((keys - key) / scale, axis=1))]
                scores = [episode(arena, shot, goal + q)[0] for q in OPTIONS]
                trials.append(
                    dict(
                        i=i,
                        donor=shot["id"],
                        score=[r["legal"] for r in scores],
                        referee=[r["events"] for r in scores],
                    )
                )
            scores = np.array([r["score"] for r in trials])
            held = scores[32:].sum(axis=0)
            curves = []
            for ix in range(4):
                probes = scores[8 * ix : 8 * ix + 8].sum(axis=0)

                def curve(order):
                    return held[decisions(probes, order)]

                choices = decisions(probes, learned)
                curves.append(
                    dict(
                        learned=held[choices].tolist(),
                        choices=choices,
                        uniform=np.mean([curve(o) for o in uniform], axis=0).tolist(),
                        shuffled=np.mean([curve(o) for o in shuffled], axis=0).tolist(),
                    )
                )
            cases.append(dict(angle=angle, trials=trials, held=held.tolist(), curves=curves))
            _write(
                OUT / "exploration-transfer.json.gz",
                dict(
                    cases=cases,
                    source_order=learned,
                    shuffled_orders=shuffled,
                    budgets=[0, 24, 40, 56, 72, 88],
                    perm_count=len(uniform),
                ),
            )
            print(
                angle,
                learned,
                {
                    name: np.mean([c[name] for c in curves], axis=0).round(2).tolist()
                    for name in ("learned", "uniform", "shuffled")
                },
                flush=True,
            )
    finally:
        arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    run(p.parse_args().feeds)
