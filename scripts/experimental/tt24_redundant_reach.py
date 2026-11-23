"""Measure transfer on new short reaches to stock-earned optical endpoints."""

import argparse
from contextlib import ExitStack, closing
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from .tt24_bounce_trial import _write, seeds
from .tt24_redundant import MarkerBounce, RedundantBounce
from .tt24_redundant_trial import conduct, markers


ROOT = Path('artifacts/tt24')
CALIBRATION = ROOT / 'redundant-game.json.gz'
OUTPUT = ROOT / 'redundant-reach.json.gz'


def move_for(effect, desired, limit=.61):
    n = len(effect)
    form = np.r_[effect.T, .008 * np.eye(n)]
    answer = lsq_linear(form, np.r_[desired, np.zeros(n)], bounds=(-limit, limit), method='bvls')
    return answer.x


def practice_goal(arena, seed, rng):
    arena.reset(seed=seed, split='held', render=False)
    a = markers(arena)
    rates = rng.normal(size=5)
    rates *= rng.uniform(.44, .60) / np.max(np.abs(rates))
    conduct(arena, rates)
    b = markers(arena)
    return None if a is None or b is None else b - a


def reach(arena, seed, action, goal):
    observation = arena.reset(seed=seed, split='held', render=False)
    before = markers(arena)
    after = conduct(arena, action)
    target = markers(arena)
    if before is None or target is None:
        return dict(command=action.tolist(), rms_pixels=None, sixth_rad=None)
    residual = (target - before - goal) * arena.height
    return dict(command=action.tolist(), rms_pixels=float(np.sqrt(np.mean(residual ** 2))),
                sixth_rad=float(abs(after['joints'][5] - observation['joints'][5])),
                measured_delta=(target - before).tolist())


def run(args):
    with gzip.open(args.calibration, 'rt') as stream:
        evidence = json.load(stream)
    effects = {name: np.asarray(row['command']) for name, row in evidence['calibration'].items()}
    flags = dict(view='none', width=192, height=144, calibration=True)
    rng = np.random.default_rng(args.seed + 11)
    with ExitStack() as stack:
        stock = stack.enter_context(closing(MarkerBounce(**flags)))
        six = stack.enter_context(closing(RedundantBounce(**flags)))
        pinned = stack.enter_context(closing(RedundantBounce(**flags, pinned=True)))
        rows = []
        for seed in seeds(args.seed, args.held):
            effect = practice_goal(stock, seed, rng)
            if effect is None:
                rows.append(dict(seed=seed, missing_stock=True))
                continue
            donor = move_for(effects['stock'], effect)
            request = donor @ effects['stock']
            choices = dict(stationary=(six, np.zeros(6)), padded=(six, np.r_[donor, 0.]),
                           learned=(six, move_for(effects['six'], request)),
                           pinned_learned=(pinned, move_for(effects['pinned'], request)))
            targets = {n: reach(arena, seed, command, effect) for n, (arena, command) in choices.items()}
            rows.append(dict(seed=seed, observed_goal=effect.tolist(), stock_intent=donor.tolist(), result=targets))
    paired = [row for row in rows if 'result' in row and all(v['rms_pixels'] is not None for v in row['result'].values())]
    if not paired:
        raise ValueError('all paired reaches lost at least one required marker')
    result = {}
    for name in paired[0]['result']:
        errors = np.array([row['result'][name]['rms_pixels'] for row in paired])
        travel = np.array([row['result'][name]['sixth_rad'] for row in paired])
        result[name] = dict(mean_pixel_error=float(errors.mean()), median_pixel_error=float(np.median(errors)),
                            within_one_pixel=int(np.sum(errors <= 1.)), median_sixth_travel_rad=float(np.median(travel)))
    report = dict(schema='tt24-redundant-reach-1', source=args.calibration.name, held=args.held, paired=len(paired), rows=rows,
                  summary=result)
    _write(args.output, report)
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, default=CALIBRATION)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--seed', type=int, default=71790462)
    parser.add_argument('--held', type=int, default=20)
    args = parser.parse_args()
    if args.held < 1:
        parser.error('held sample count must be positive')
    run(args)


if __name__ == '__main__':
    main()
