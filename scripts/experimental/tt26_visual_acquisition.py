"""Learn which anonymous movement returns which incoming ball from legal landings alone."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt25_pose_demonstrations import episode
from .tt26_grip_course import OUT, PRIOR, Grip, ReferenceReceiver, load_match
from .tt26_receiving_speed import draw


class VisualRewardMemory:
    def __init__(self, count):
        self.n = count
        self.keys = []
        self.selected = []
        self.legal = []
        self.scale = np.tile([36, 36, 9, 9, 9, 9, 4], 2)

    def choose(self, key, order, explore):
        if not self.keys:
            return int(order[0])
        distance = np.linalg.norm((np.asarray(self.keys) - key) / self.scale, axis=1)
        weights = np.exp(-0.5 * (distance / 0.85) ** 2)
        attempts = np.bincount(self.selected, weights=weights, minlength=self.n)
        wins = np.bincount(self.selected, weights=weights * self.legal, minlength=self.n)
        score = (wins + 0.20) / (attempts + 1)
        if explore:
            score += 0.35 / (attempts + 1)
        return int(order[np.argmax(score[order])])

    def update(self, key, choice, legal):
        self.keys.append(key)
        self.selected.append(choice)
        self.legal.append(legal)


def run(path):
    shots, (_, goals, _) = load_match(path)
    ids = _read(PRIOR / "recorded-nearby-lessons.json.gz")["donors"]
    arena = ReferenceReceiver()
    grip = Grip(arena.model)
    records = []
    try:
        for src in _read(OUT / "grip-source.json.gz")["records"]:
            grip.set(src["angle"])
            for seed in (5113961, 1961753):
                rng = np.random.default_rng(seed)
                # Anonymous actions are six postures acquired in earlier nominal matches.
                # The visual signatures, identities and return labels are deliberately discarded.
                actions = goals[rng.permutation(len(goals))]
                learners = {
                    name: (bias, VisualRewardMemory(len(goals)))
                    for name, bias in (
                        ("bounce", src["selected"]),
                        ("cold", 0.0),
                        ("shuffled", src["shuffled"]),
                    )
                }
                rows = []
                for t in range(144):
                    shot = draw(shots, ids, rng, t)
                    _, key = episode(arena, shot, collect=True)
                    order = rng.permutation(len(goals))
                    results = {}
                    for name, (bias, memory) in learners.items():
                        choice = memory.choose(key, order, t < 96)
                        goal = actions[choice].copy()
                        goal[4] += bias
                        score = episode(arena, shot, goal)[0]
                        if t < 96:
                            memory.update(key, choice, score["legal"])
                        results[name] = dict(choice=choice, legal=score["legal"], referee=score["events"])
                    rows.append(dict(tick=t, donor=shot["id"], key=key.tolist(), results=results))
                records.append(dict(angle=src["angle"], seed=seed, rows=rows))
                _write(OUT / "visual-acquisition.json.gz", dict(records=records))
                print(
                    src["angle"],
                    seed,
                    {
                        k: [
                            sum(r["results"][k]["legal"] for r in rows[i : i + 24])
                            for i in range(0, 144, 24)
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
