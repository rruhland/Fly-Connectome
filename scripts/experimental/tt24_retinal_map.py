"""Fit a local continuous command decoder from wrist motion and joint displacements."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .tt24_effects import excitation, transmission
from .tt24_bounce import BounceArena
from .tt24_effects import MotorEffects
from .tt24_remap_game import run_policy


def visual_motion(first, second):
    """Estimate mean two-dimensional motion in each of nine retinal regions."""
    # Color gradients preserve boundaries between surfaces of equal luminance.
    first, second = (x.astype(float) / 255. for x in (first, second))
    average = (first + second) * .5
    gy, gx = np.gradient(average, axis=(0, 1))
    diff = second - first
    rows = []
    for yy in np.array_split(np.arange(2, first.shape[0]-2), 3):
        for xx in np.array_split(np.arange(2, first.shape[1]-2), 3):
            grid = np.ix_(yy, xx)
            design = np.stack((gx[grid], gy[grid]), axis=-1).reshape(-1, 2)
            target = -diff[grid].ravel()
            # One robust refit downweights isolated occlusions or large motions.
            xy = np.linalg.solve(design.T @ design + np.eye(2)*.03, design.T @ target)
            weight = 1. / (1. + ((target - design @ xy)/.045)**2)
            weighted = design.T * weight
            rows.extend(np.linalg.solve(weighted @ design + np.eye(2)*.03, weighted @ target))
    return np.asarray(rows)


def calibrate(arena, wiring, seed, regime):
    ticks = round(.1 / arena.dt)
    schedule = excitation(np.random.default_rng(seed), passes=2)
    model = MotorEffects(5, blocks=len(schedule))
    rows, features = [], []
    done = True
    frames = drops = 0
    for probe in schedule:
        while True:
            if done:
                obs = arena.reset(seed=seed + drops, split='practice', regime=regime)
                drops += 1
                previous = np.zeros(5)
            start = obs['joints'].copy()
            flow = np.zeros(18)
            for tick in range(ticks * 2):
                command = probe if tick < ticks else np.zeros(5)
                new, _, done = arena.step(wiring @ command)
                flow += visual_motion(obs['rgb'], new['rgb'])
                obs = new
                frames += 1
                if done:
                    break
            if tick == ticks * 2 - 1:
                delta = obs['joints'] - start
                rows.append(probe.copy())
                features.append(flow)
                model.update(probe, delta, previous)
                previous = delta
                break
    x, y = np.array(rows), np.array(features)
    optic = np.linalg.lstsq(x, y-y.mean(axis=0), rcond=.01)[0].T
    return np.vstack([model.coefficients()[:5].T, optic]), dict(frames=frames, drops=drops)


def decoder(source, target, visual_weight):
    # Relative scaling allows the prior to remain useful with different camera focal lengths.
    def normalize(m):
        amount = np.sqrt(np.mean(m * m, axis=1))
        floor = np.median(amount) * .25 + 1e-6
        return m / np.maximum(floor, amount)[:, None] / np.sqrt(len(m))
    source = np.vstack((normalize(source[:5]), visual_weight*normalize(source[5:])))
    target = np.vstack((normalize(target[:5]), visual_weight*normalize(target[5:])))
    return np.linalg.lstsq(target, source, rcond=.04)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=381931)
    parser.add_argument('--starts', type=int, default=10)
    parser.add_argument('--changed-physics', action='store_true')
    args = parser.parse_args()
    record = json.loads(gzip.decompress(args.checkpoint.read_bytes()))
    rng = np.random.default_rng(args.seed)
    starts = rng.integers(1000000, 2**31-1, args.starts).tolist()
    wire = transmission(rng)
    eye = np.eye(5)
    settings = dict(view='wrist', width=record.get('width', 96), height=record.get('height', 72))
    src = BounceArena(**settings)
    dst = BounceArena(**settings, arm='alternate')
    regime = 'changed' if args.changed_physics else 'standard'
    try:
        source, cost1 = calibrate(src, eye, args.seed, 'standard')
        target, cost2 = calibrate(dst, wire, args.seed + 403, regime)
        result = {}
        for label, arena, matrix, transform, law in (
            ('stock', src, eye, eye, 'standard'), ('alternate', dst, eye, eye, regime),
            ('mixed', dst, wire, eye, regime),
            ('encoder', dst, wire, decoder(source, target, 0.), regime),
            ('wrist_encoder', dst, wire, decoder(source, target, .6), regime),
            ('known_wiring', dst, wire, np.linalg.inv(wire), regime),
            ('hold', dst, eye, eye, regime), ('blind', dst, wire, decoder(source, target, .6), regime)):
            result[label] = [run_policy(arena, s, record, transform, matrix, regime=law,
                                        hold=label=='hold', blind=label=='blind') for s in starts]
            print(label, [r['rebounds'] for r in result[label]], flush=True)
        data = dict(checkpoint=args.checkpoint.name, starts=starts, physics=regime, source_cost=cost1,
                    target_cost=cost2, source=source.tolist(), target=target.tolist(), wiring=wire.tolist(),
                    outcomes=result)
        args.output.write_text(json.dumps(data, separators=(',', ':')) + '\n')
    finally:
        src.close()
        dst.close()


if __name__ == '__main__':
    main()
