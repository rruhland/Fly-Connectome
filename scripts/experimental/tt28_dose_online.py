"""Compare regulation reward learning from the frozen zero, short and long solo checkpoints."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _write
from .tt25_pose_demonstrations import episode
from .tt26_grip_course import Grip, ReferenceReceiver, load_match
from .tt26_receiving_speed import draw
from .tt27_flight_feedback import OUT as BANK_DIR
from .tt27_flight_feedback import load_bank
from .tt28_dose import MOUNTS, OUTPUT, target_banks, target_summary

DECK_SEED = 28118901
EXPLORE_SEED = 28712056
PRACTICE = 56
HELD = 48
DOSES = ("0", "16", "192")
DISPLACEMENTS = np.array([
    [0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0.14],
    [0., 0., 0., 0., -0.14],
    [0., 0., 0., 0.11, 0.],
    [0., 0., 0., -0.11, 0.],
    [0., 0.07, -0.11, 0.04, 0.],
    [0., -0.07, 0.11, -0.04, 0.],
])


class Outcomes:
    """Update one camera-neighborhood reward estimate after the executed action."""

    def __init__(self):
        self.keys = []
        self.actions = []
        self.rewards = []

    def choose(self, query):
        trials = np.full(len(DISPLACEMENTS), 4.)
        wins = np.full(len(DISPLACEMENTS), .75)
        wins[0] = 1.5
        if self.keys:
            distance = np.linalg.norm(np.asarray(self.keys) - query, axis=1)
            similarity = .20 + .80 * np.exp(-.5 * (distance / 1.8) ** 2)
            trials += np.bincount(self.actions, weights=similarity, minlength=len(DISPLACEMENTS))
            wins += np.bincount(self.actions, weights=similarity * self.rewards, minlength=len(DISPLACEMENTS))
        return int(np.argmax(wins / trials))

    def update(self, query, action, legal):
        self.keys.append(query.copy())
        self.actions.append(action)
        self.rewards.append(legal)


def run(args):
    label, feeds, output = args
    began = perf_counter()
    source = output / f"dose-checkpoints-{label}.npz"
    shots, (_, _, scales) = load_match(feeds)
    bank = load_bank()["rows"]
    donors = [r["id"] for r in bank]
    keys = np.array([r["key"] for r in bank])
    goals = np.array([r["goal"] for r in bank])
    with np.load(source, allow_pickle=False) as checkpoint:
        if checkpoint["schema"].item() != "tt28-solo-dose-checkpoint-1":
            raise ValueError("Unknown solo checkpoint format")
        snaps = {str(d): dict(rotation=r) for d, r in zip(
            checkpoint["doses"], checkpoint["rotation"], strict=True) if str(d) in DOSES}
    if set(snaps) != set(DOSES):
        raise ValueError("The solo checkpoint must contain zero, short and long doses")
    arena = ReferenceReceiver()
    fixture = Grip(arena.model)
    fixture.set(*MOUNTS[label])
    banks = target_banks(arena, goals, snaps)
    rng = np.random.default_rng(DECK_SEED)
    exploration = np.random.default_rng(EXPLORE_SEED)
    learners = {d: Outcomes() for d in DOSES}
    rows = []
    replay_count = 0
    try:
        for i in range(PRACTICE + HELD):
            practicing = i < PRACTICE
            shot = draw(shots, donors, rng, i)
            blank, key = episode(arena, shot, collect=True)
            explore = None
            if i < len(DISPLACEMENTS):
                explore = i
            elif practicing and exploration.random() < .28:
                explore = int(exploration.integers(len(DISPLACEMENTS)))
            results = {}
            performed = {}
            for dose, model in learners.items():
                if key is None:
                    results[dose] = dict(action=None, **blank)
                    results[f"frozen_{dose}"] = dict(action=None, **blank)
                    continue
                idx = np.argmin(np.linalg.norm((keys - key) / scales, axis=1))
                q = banks[dose][idx]
                action = model.choose(key / scales) if explore is None else explore
                for mode, act in ((dose, action), (f"frozen_{dose}", 0)):
                    goal = q + DISPLACEMENTS[act]
                    as_key = tuple(goal)
                    if as_key not in performed:
                        result, _ = episode(arena, shot, goal=goal)
                        if result["legal"]:
                            if result != episode(arena, shot, choose_goal=lambda seen: goal)[0]:
                                raise ValueError(f"Online target failed replay for {label}/{i}/{mode}")
                            replay_count += 1
                        performed[as_key] = result
                    results[mode] = dict(action=act, **performed[as_key])
                if practicing:
                    model.update(key / scales, action, results[dose]["legal"])
            if i < 8:
                group = "practice_1_8"
            elif i < 24:
                group = "practice_9_24"
            elif practicing:
                group = "practice_25_56"
            else:
                group = "held_57_104"
            rows.append(dict(attempt=i + 1, group=group, shot=shot, shared_exploration=explore,
                             camera_key=None if key is None else key.tolist(),
                             physical_actions=len(performed), results=results))
    finally:
        arena.close()
    summary = target_summary(rows)
    for group, stats in summary.items():
        score = [r["results"] for r in rows if r["group"] == group]
        for dose in DOSES:
            stats[dose]["gains_vs_own_frozen"] = sum(r[dose]["legal"] > r[f"frozen_{dose}"]["legal"]
                                                    for r in score)
            stats[dose]["losses_vs_own_frozen"] = sum(r[dose]["legal"] < r[f"frozen_{dose}"]["legal"]
                                                     for r in score)
    _write(output / f"dose-online-{label}.json.gz",
           dict(schema="tt28-online-dose-1", tool=label, solo_checkpoint_sha256=_hash(source),
                recorded_launch_sha256=_hash(feeds), bank_sha256=_hash(BANK_DIR / "demonstrations.json.gz"),
                deck_seed=DECK_SEED, exploration_seed=EXPLORE_SEED, practice_shots=PRACTICE,
                held_shots=HELD, displacements=DISPLACEMENTS.tolist(), rows=rows,
                input="four stereo RGB frames, issued requests, own past chosen action/legal outcome",
                held_labels="separately drawn perturbations of the 19 historical training donors",
                successful_unique_replays=replay_count, summary=summary, wall_seconds=perf_counter() - began))
    print(label, {group: {d: stats[d]["legal"] for d in stats} for group, stats in summary.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
        list(pool.map(run, [(label, args.feeds, args.output) for label in MOUNTS]))
