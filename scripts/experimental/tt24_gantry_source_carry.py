"""Audit frozen Cartesian-source readouts on the physical SO-101 simulation."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_bounce_ident import CleanState
from .tt24_bounce_ident_transfer import _verify_tape
from .tt24_world_diagnostic import PhysicalInverse
from .tt24_world_prior import CHECKPOINT, WorldPrior


def trace(arena, seed, prior, regime):
    obs = arena.reset(seed=seed, split='held', regime=regime, render=False)
    if prior is not None:
        state = CleanState(arena)
        inverse = PhysicalInverse(arena, obs['joints'])
    commands = []
    for _ in range(arena.max_steps):
        command = inverse.target(prior.target(state.observe(obs))) if prior is not None else np.zeros(5)
        commands.append(command.tolist())
        obs, _, finished = arena.step(command, render=False)
        if finished:
            break
    return arena.judge_metrics() | dict(commands=commands)


def summary(episodes):
    runs = [row['longest_consecutive_rebounds'] for row in episodes]
    return dict(mean_rebounds=float(np.mean([row['rebounds'] for row in episodes])),
                mean_consecutive=float(np.mean(runs)), max_consecutive=max(runs),
                at_least_ten=sum(count >= 10 for count in runs), consecutive_counts=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/tt24'))
    parser.add_argument('--analytic', type=Path, default=CHECKPOINT)
    parser.add_argument('--gantry', type=Path, default=Path('artifacts/tt24/gantry-source.npz'))
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--horizon', type=int, default=450)
    parser.add_argument('--seed', type=int, default=51738174)
    parser.add_argument('--arm', choices=('stock', 'alternate'), default='stock')
    args = parser.parse_args()
    if min(args.count, args.horizon) < 1:
        parser.error('need positive count and horizon')
    args.out.mkdir(parents=True, exist_ok=True)
    arena = BounceArena(view='none', arm=args.arm, max_steps=args.horizon)
    seeds = np.random.default_rng(args.seed).integers(1, 2**31 - 1, args.count).tolist()
    regime = 'standard' if args.arm == 'stock' else 'changed'
    names = {'analytic_prior': args.analytic, 'gantry_refined': args.gantry}
    hashes = {mode: sha256(path.read_bytes()).hexdigest() for mode, path in names.items()}
    controls = {mode: WorldPrior.load(path) for mode, path in names.items()} | {'stationary': None}
    cases = {}
    try:
        for mode, model in controls.items():
            rows = [trace(arena, seed, model, regime) for seed in seeds]
            strongest = max(rows, key=lambda row: row['longest_consecutive_rebounds'])
            _verify_tape(arena, strongest, regime)
            cases[mode] = dict(summary=summary(rows), exact_replay_seed=strongest['seed'], episodes=rows)
            print(args.arm, mode, cases[mode]['summary'], flush=True)
    finally:
        arena.close()
    assert hashes == {mode: sha256(path.read_bytes()).hexdigest() for mode, path in names.items()}
    report = dict(schema='tt24-cartesian-carry-audit-1', arm=args.arm, regime=regime, horizon=args.horizon,
                  seeds=seeds, hashes=hashes, target_training=False,
                  privileged_test_state=True, privileged_jaw_differential=True, outcomes=cases)
    with gzip.open(args.out / f'gantry-source-arm-{args.arm}.json.gz', 'wt', encoding='utf-8') as file:
        json.dump(report, file, separators=(',', ':'))


if __name__ == '__main__':
    main()
