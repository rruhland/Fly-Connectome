"""Online reward learning on reachable full-size feeds after solo tool practice."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt25_pose_demonstrations import episode
from .tt26_grip_course import Grip, OUT, PRIOR, ReferenceReceiver, SETTINGS, load_match


class RewardLearner:
    def __init__(self, setting):
        self.wins = np.ones(len(SETTINGS))
        self.trials = np.full(len(SETTINGS), 3.0)
        preferred = int(np.flatnonzero(SETTINGS == setting)[0])
        self.wins[preferred] += 3.0
        self.trials[preferred] += 3.0

    def best(self):
        return int(np.argmax(self.wins / self.trials))

    def update(self, action, legal):
        self.wins[action] += int(legal)
        self.trials[action] += 1.0


def draw(shots, ids, rng, tick):
    shot = shots[ids[int(rng.integers(len(ids)))]].copy()
    # Reachable curriculum from real launches, with distinct continuous perturbations.
    for prefix, sigma in (("pos", 0.008), ("vel", 0.035), ("w_vel", 1.5)):
        for axis, jitter in zip("xyz", rng.normal(0, sigma * (1 + tick % 3), 3), strict=True):
            shot[f"{prefix}_{axis}"] += jitter
    return shot


def run(path):
    shots, (keys, goals, scale) = load_match(path)
    donors = _read(PRIOR / "recorded-nearby-lessons.json.gz")["donors"]
    arena = ReferenceReceiver()
    grip = Grip(arena.model)
    records = []
    try:
        for row in _read(OUT / "grip-source.json.gz")["records"]:
            grip.set(row["angle"])
            for seed in (193981, 571817, 211919, 441130):
                schedule = np.random.default_rng(seed)
                choices = np.random.default_rng(seed + 3).permutation(np.tile(np.arange(len(SETTINGS)), 4))
                learners = {
                    name: RewardLearner(setting)
                    for name, setting in (
                        ("bounce", row["selected"]),
                        ("cold", 0.0),
                        ("shuffled", row["shuffled"]),
                    )
                }
                history = []
                for tick in range(80):
                    shot = draw(shots, donors, schedule, tick)
                    _, key = episode(arena, shot, collect=True)
                    origin = int(np.argmin(np.linalg.norm((keys - key) / scale, axis=1)))
                    scores = {}
                    step = {}
                    for name, learner in learners.items():
                        # Same forced probes, every fourth training trial. Final 16 are frozen.
                        index = int(choices[tick // 4]) if tick < 64 and tick % 4 == 3 else learner.best()
                        if index not in scores:
                            action = goals[origin].copy()
                            action[4] += SETTINGS[index]
                            scores[index] = episode(arena, shot, goal=action)[0]
                        result = scores[index]
                        step[name] = dict(
                            setting=int(index),
                            legal=result["legal"],
                            contact=result["contact"],
                            referee=result["events"],
                        )
                        if tick < 64:
                            learner.update(index, result["legal"])
                    history.append(dict(tick=tick, id=shot["id"], results=step))
                records.append(
                    dict(
                        angle=row["angle"],
                        seed=seed,
                        history=history,
                        final={k: v.best() for k, v in learners.items()},
                    )
                )
                _write(OUT / "receiving-speed.json.gz", dict(records=records, settings=SETTINGS.tolist()))
                print(
                    row["angle"],
                    seed,
                    {
                        k: [
                            sum(x["results"][k]["legal"] for x in history[i : i + 16])
                            for i in range(0, 80, 16)
                        ]
                        for k in learners
                    },
                    flush=True,
                )
    finally:
        arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    run(p.parse_args().feeds)
