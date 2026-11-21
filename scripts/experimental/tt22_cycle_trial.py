"""Search and verify uninterrupted two-robot legal table-tennis rallies."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt22_cycle import ARENA, CyclePlanner, World, rally


OUTPUT = ROOT / 'artifacts/tt22'
DEVELOP = (2059027790, 746909458, 1634221833)
HELD = (1909157674, 728025937, 530614602, 1413200957)


def verify(rows):
    world = World()
    try:
        for row in rows:
            world.reset(row['seed'])
            for near, far in row['actions']:
                _, reward, _, _, _ = world.step(near, far)
            if (world.arena.point_result() != row['result'] or reward != row['reward']
                    or json.dumps(world.arena.events) != json.dumps(row['events'])):
                raise ValueError('physical rally replay changed')
    finally:
        world.close()
    return len(rows)


def run(path, *, seeds, height, speed, verbose):
    world = World()
    planner = CyclePlanner(world, height=height, speed=speed)
    began = perf_counter()
    rows = []
    try:
        for seed in seeds:
            row = rally(world, planner, seed, verbose=verbose)
            rows.append(row)
            result = dict(schema='tt22-physical-cycle-1', mode='privileged', settings=ARENA,
                          envelope={'depth': .91, 'lateral': 0., 'height': height, 'speed': speed},
                          learned_policy=False, feed='one ordinary rival serve then uninterrupted simulated point',
                          seed_schedule=seeds, rows=rows, verified=verify(rows), seconds=perf_counter()-began,
                          implementation={name:_hash(Path(__file__).with_name(name)) for name in
                                          ('tt22_cycle.py','tt22_cycle_trial.py')})
            _write(path, result)
            print(seed, row['legal_alternating_returns'], row['result'], flush=True)
    finally:
        planner.close()
        world.close()
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--held', action='store_true')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--height', type=float, default=.205)
    parser.add_argument('--speed', type=float, default=2.60)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    seeds = args.seeds or (HELD if args.held else DEVELOP)
    path = args.output or OUTPUT/('cycle-held.json.gz' if args.held else 'cycle-development.json.gz')
    path.parent.mkdir(exist_ok=True)
    run(path, seeds=seeds, height=args.height, speed=args.speed, verbose=args.verbose)
