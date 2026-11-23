"""Learn continuous local feedback corrections from whole-episode returns.

Each stored correction has a sensory center, a width and a five-axis drive.
There are no time-indexed commands. Replaying matching release seeds during
development permits causal comparisons and is charged as target experience.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_motor import OpticalMemory


def episode(arena, start, memory, *, split, blind=False):
    optic = OpticalMemory(retinal='absolute')
    obs = arena.reset(seed=int(start), split=split, render=not blind)
    home = obs['joints'].copy()
    last = np.zeros(5)
    features, events = [], []
    score = frame = 0
    done = False
    if memory:
        centers, bandwidths, impulses = map(np.array, zip(*memory))
    while not done:
        if frame % 2 == 0:
            view = optic.observe(None if blind else obs['rgb'])
            # All terms are immediate optical or proprioceptive measurements.
            feat = np.r_[view[1], view[2:5], 2*(view[2:5] - view[8:11]),
                         .8*view[5:8], 2*(obs['joints'] - home), .18*obs['joint_velocity']]
            features.append(feat)
            if memory:
                distance = np.sum((centers - feat)**2, axis=1)/bandwidths**2
                drive = np.exp(-.5*distance) @ impulses
            else:
                drive = np.zeros(5)
            last = .30 * last + .70 * np.clip(drive, -.65, .65)
        action = np.clip(last - 1.8*(obs['joints'] - home) - .035*obs['joint_velocity'], -1., 1.)
        obs, reward, done = arena.step(action, render=not blind and frame % 2 == 1)
        if reward:
            score += reward
            events.append(frame//2)
        frame += 1
    return dict(rebounds=arena.judge_metrics()['rebounds'], score=float(score), ticks=frame), np.array(features), events


def cost(records):
    # Favor useful lower-tail improvements across practice episodes and then total reward.
    val = np.array([r['score'] for r in records])
    return float(np.mean(val) + .3*np.min(val))


def experiment(args):
    rng = np.random.default_rng(args.seed)
    starts = rng.integers(1, 2**31-1, args.contexts).tolist()
    held = np.random.default_rng(args.held_seed).integers(1, 2**31-1, args.held).tolist()
    arena = BounceArena(view='wrist', width=64, height=48, apex_reward_weight=args.apex)
    memory = []
    exposure = []
    try:
        def run(mem, ss, split, blind=False):
            batch = [episode(arena, x, mem, split=split, blind=blind) for x in ss]
            scores, traces, events = zip(*batch)
            return list(scores), traces, events

        scores, traces, events = run(memory, starts, 'practice')
        exposure.append(dict(accepted=True, score=scores))
        for k in range(args.steps):
            if k % 2 == 0:
                j = int(rng.integers(len(starts)))
                n = len(traces[j])
                if events[j] and rng.random() < .65:
                    # Explore near experiences leading up to delayed rewards.
                    idx = max(0, int(rng.choice(events[j])) - int(rng.integers(1, 9)))
                else:
                    idx = int(rng.integers(n))
                center = traces[j][idx].copy()
                width = rng.uniform(.30, 1.1)
                direction = rng.normal(size=5)
                direction *= rng.uniform(.15, .72) / np.linalg.norm(direction)
                parent = memory.copy()
            mutation = (center, width, direction if k%2==0 else -direction)
            candidate = parent + [mutation]
            new, tr, ev = run(candidate, starts, 'practice')
            better = cost(new) > cost(scores) + 1e-6
            if better:
                scores, traces, events = new, tr, ev
                memory = candidate
                print('practice', (k+2)*len(starts), len(memory), [r['rebounds'] for r in new], flush=True)
            exposure.append(dict(accepted=better, score=new))
        results = {}
        for label, mem, blind in [('live', memory, False), ('blind', memory, True), ('zero', [], False)]:
            result, _, _ = run(mem, held, 'held', blind=blind)
            results[label] = result
            print(label, [r['rebounds'] for r in result], flush=True)
        out = dict(seed=args.seed, practice_starts=starts, held_starts=held, apex_weight=args.apex,
                   exposure=exposure, held=results,
                   memory=[[c.tolist(), w, v.tolist()] for c,w,v in memory])
        with gzip.open(args.out, 'wt') as stream:
            json.dump(out, stream, separators=(',', ':'))
    finally:
        arena.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--seed', type=int, default=41397)
    p.add_argument('--held-seed', type=int, default=332898)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--contexts', type=int, default=2)
    p.add_argument('--held', type=int, default=10)
    p.add_argument('--apex', type=float, default=.25)
    experiment(p.parse_args())
