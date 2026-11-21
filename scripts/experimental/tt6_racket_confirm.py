"""Confirm the racket experiment at a second calibrated native hall station."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _counts, _hash, _point, _read, _write
from .tt5_checkpoint import load, setup
from .tt6_racket import RacketAdaptation, RacketPlayer, set_racket_response
from .tt6_racket_trial import summarize
from .tt6_render import cache_hall_stereo


RECEIPT = ROOT / 'artifacts/tt5/first-person-source.json.gz'
CHECKPOINT = ROOT / 'artifacts/tt6/second-station.npz'
MATCH = (('stock', 3), ('very_soft', 4), ('stock', 3), ('very_soft', 4), ('soft', 4))


def _run(job):
    condition, gain, schedule, cached, reacquisition = job
    _, _, wiring = setup(RECEIPT, 1)
    arena = wiring.arena
    try:
        if cached:
            cache_hall_stereo(arena)
        station = load(CHECKPOINT)
        learner = RacketPlayer(station, RacketAdaptation(gain=gain, default_speed=station.stroke_speed,
                                                         reacquisition=reacquisition))
        rows = []
        for phase, offset, response, seed in schedule:
            set_racket_response(arena, response)
            row = _point(wiring, learner, seed)
            rows.append(dict(phase=phase, offset=offset, racket=response, **row, **_counts(row)))
        return condition, rows
    finally:
        arena.close()


def main(args):
    practice = _read(args.practice)['mini_practice']
    rng = np.random.default_rng(args.seed)
    schedule = [[phase, offset, response, int(rng.integers(0, 2**31 - 1))]
                for phase, (response, points) in enumerate(MATCH) for offset in range(points)]
    gains = dict(varied_mini=practice['varied']['gain'], ordinary_mini=practice['ordinary']['gain'], generic_fast=.80)
    output = dict(schema='tt6-racket-independent-confirmation-v1', seed=args.seed, schedule=schedule, gains=gains,
                  installation=1, checkpoint_sha256=_hash(CHECKPOINT), receipt_sha256=_hash(RECEIPT),
                  practice_sha256=_hash(args.practice), points_per_group=practice['points_per_group'], scores={},
                  renderer='cached', native_prefix={}, reacquisition=args.reacquisition,
                  implementation_sha256={path.name: _hash(path) for path in
                                         (Path(__file__), Path(__file__).with_name('tt6_racket.py'))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run, (condition, gain, schedule, True, args.reacquisition))
                   for condition, gain in gains.items()]
        for future in as_completed(futures):
            condition, rows = future.result()
            output['scores'][condition] = rows
            output['summary'] = summarize(output['scores'])
            _write(args.output, output)
            print(condition, output['summary'][condition], flush=True)
        if not args.native_prefix:
            return
        futures = [executor.submit(_run, (condition, gains[condition], schedule[:args.native_prefix], False,
                                          args.reacquisition))
                   for condition in ('varied_mini', 'ordinary_mini')]
        for future in as_completed(futures):
            condition, rows = future.result()
            comparisons = []
            for native, cached in zip(rows, output['scores'][condition][:args.native_prefix], strict=True):
                comparisons.append(dict(seed=native['seed'], native=native, cached_sha256=cached['action_sha256'],
                                        same_actions=native['action_sha256'] == cached['action_sha256'],
                                        same_score=native['evaluator'] == cached['evaluator']))
            output['native_prefix'][condition] = comparisons
            _write(args.output, output)
            print(condition, [(r['same_actions'], r['same_score']) for r in comparisons], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('practice', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=92817041)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--reacquisition', action='store_true')
    parser.add_argument('--native-prefix', type=int, choices=range(sum(n for _, n in MATCH) + 1), default=0)
    main(parser.parse_args())
