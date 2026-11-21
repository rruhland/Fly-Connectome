"""Find a visual demonstration phase and execute a short joint-target sequence."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt22_rally_trial import RallyArena, opponent
from .tt22_visual_apprentice import Apprentice, COUNT, VisualFeatures, own_features


class SequencePlayer(Apprentice):
    def __init__(self, dataset, receipts, vision=True):
        super().__init__(dataset)
        ordered = []
        for row in receipts["rows"]:
            acts = np.array(row["actions"])
            ordered.append(np.cumsum(acts[:, :5], axis=0))
        self.targets = np.vstack(ordered)
        self.starts = np.r_[True, np.diff(dataset["points"]) != 0]
        self.ends = np.r_[np.flatnonzero(self.starts)[1:] - 1, len(self.targets) - 1]
        self.eyes = vision
        self.reset()

    def reset(self):
        self.cost = np.where(self.starts, 0.0, 1e4)
        self.queue = []

    def action(self, vision, body):
        d = np.mean((self.b - body / self.body_scale) ** 2, axis=1)
        if self.eyes:
            v = np.mean((self.v - vision / self.vscale).reshape(-1, COUNT, 10) ** 2, axis=2)
            d += v @ self.weight
        self.cost += np.minimum(d, 10.0)
        idx = np.argmin(self.cost)
        end = self.ends[np.searchsorted(self.ends, idx)]
        window = self.targets[np.minimum(np.arange(idx, idx + 4), end)]
        # Pool the short, overlapping future targets from recent visual matches.
        self.queue = (self.queue + [window])[-4:]
        proposals = [a[len(self.queue) - 1 - k] for k, a in enumerate(self.queue)]
        target = np.average(proposals, axis=0, weights=np.arange(1, len(proposals) + 1))
        normalized = 0.86 * (self.cost - self.cost.min())
        # Most transitions advance one frame. Small timing shifts and new matches remain possible.
        forward = np.r_[1e4, normalized[:-1]]
        forward[self.starts] = 1e4
        two_step = np.r_[1e4, forward[:-1]] + 0.08
        two_step[self.starts] = 1e4
        self.cost = np.minimum.reduce([forward, two_step, normalized + 0.08, np.full(len(d), 1.8)])
        return np.r_[np.clip(target - body[:5], -1.0, 1.0), 0.0]


def run(args):
    dataset = np.load(args.dataset)
    receipts = _read(args.demonstrations)
    player = SequencePlayer(dataset, receipts, vision=not args.blind)
    arena = RallyArena(receipts["spread"], receipts["pace"])
    opponent(arena, receipts["rival"])
    seeds = np.random.default_rng(args.seed).integers(2**31, size=args.count).tolist()
    output = dict(training=[r["seed"] for r in receipts["rows"]], blind=args.blind, rows=[])
    if set(output["training"]) & set(seeds):
        raise ValueError("reused demonstration starts")
    try:
        for seed in seeds:
            arena.reset(seed=seed, server=-1, render=False)
            featurizer = VisualFeatures()
            player.reset()
            target = np.zeros(6)
            action = np.zeros(6)
            commands = []
            for tick in range(arena.max_steps):
                features = featurizer.observe(arena.render_cameras()["left"]) if not args.blind else None
                action = player.action(features, own_features(target, action))
                target += action
                commands.append(action.tolist())
                _, _, done, capped, _ = arena.step(action, render=False)
                if done or capped:
                    break
            outcome = arena.point_result()
            output["rows"].append(
                dict(seed=seed, outcome=outcome, actions=commands, events=arena.events.copy())
            )
            _write(args.output, output)
            print(seed, outcome["legal_returns"], outcome["rival_returns"], flush=True)
    finally:
        arena.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("demonstrations", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--seed", type=int, default=273146)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--blind", action="store_true")
    run(ap.parse_args())
