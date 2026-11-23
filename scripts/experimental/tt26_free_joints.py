"""Choose any of five joint adjustments using only rewards from varied wrist-camera drops."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import CarriedWristPolicy
from .tt25_pose_demonstrations import episode
from .tt26_grip_course import Grip, OUT, PRIOR, ReferenceReceiver, bounce, load_match
from .tt26_receiving_speed import draw

OPTIONS = np.vstack((np.zeros(5), 0.24 * np.eye(5), -0.24 * np.eye(5)))


def practice():
    arena = BounceArena(view="wrist", width=192, height=144, max_steps=360)
    policy = CarriedWristPolicy()
    grip = Grip(arena._model)
    records = []
    try:
        for mount in (-0.22, 0.22, 0.0):
            grip.set(mount)
            rng = np.random.default_rng(138104210)
            rows = []
            for repeat in range(10):
                order = rng.permutation(len(OPTIONS))
                for index in order:
                    seed = int(rng.integers(2**30))
                    regime = ("standard", "changed")[repeat % 2]
                    stats = bounce(arena, policy, seed, OPTIONS[index], "practice", regime)
                    rows.append(dict(index=int(index), seed=seed, **stats))

            def select(rewards):
                return int(np.argmax(np.bincount([r["index"] for r in rows], weights=rewards)))

            reward = [r["reward"] for r in rows]
            selected = select(reward)
            shuffled = select(rng.permutation(reward))
            metrics = []
            for i in range(20):
                seed = int(rng.integers(2**30))
                regime = ("standard", "changed")[i % 2]
                results = {
                    str(j): bounce(arena, policy, seed, OPTIONS[j], "held", regime)
                    for j in {0, selected, shuffled}
                }
                metrics.append(dict(seed=seed, results=results))
            record = dict(mount=mount, practice=rows, selected=selected, shuffled=shuffled, held=metrics)
            records.append(record)
            _write(OUT / "five-joint-practice.json.gz", dict(records=records, options=OPTIONS.tolist()))
            print(
                mount,
                selected,
                shuffled,
                [sum(x["results"][str(j)]["streak"] for x in metrics) for j in (selected, 0, shuffled)],
                flush=True,
            )
    finally:
        arena.close()


def receiving(path):
    shots, (keys, goals, scales) = load_match(path)
    donors = _read(PRIOR / "recorded-nearby-lessons.json.gz")["donors"]
    arena = ReferenceReceiver()
    grip = Grip(arena.model)
    records = []
    try:
        for record in _read(OUT / "five-joint-practice.json.gz")["records"]:
            grip.set(record["mount"])
            rng = np.random.default_rng(315819511)
            settings = (record["selected"], 0, record["shuffled"])
            rows = []
            for i in range(72):
                shot = draw(shots, donors, rng, i)
                _, key = episode(arena, shot, collect=True)
                goal = goals[np.argmin(np.linalg.norm((keys - key) / scales, axis=1))]
                results = {str(j): episode(arena, shot, goal + OPTIONS[j])[0] for j in set(settings)}
                rows.append(dict(tick=i, donor=shot["id"], results=results))
            records.append(dict(mount=record["mount"], settings=settings, rows=rows))
            _write(OUT / "five-joint-match.json.gz", dict(records=records, seed=315819511))
            print(
                record["mount"],
                settings,
                [sum(x["results"][str(j)]["legal"] for x in rows) for j in settings],
                flush=True,
            )
    finally:
        arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("part", choices=("practice", "match"))
    p.add_argument("--feeds", type=Path)
    args = p.parse_args()
    if args.part == "practice":
        practice()
    else:
        receiving(args.feeds)
