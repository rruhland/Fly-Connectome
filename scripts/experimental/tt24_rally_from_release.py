"""Fit the same feedback law from release and test it without a supplied opening."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt23_sequence import HANDOVER, REPLICATION, course, verify
from .tt24_rally_policy import JointPolicy
from .tt24_rally_trial import (COACHED, DATA, OUTPUT, STEP_SIZE, Station, archive_schedule,
                               demo_schedule, difference_count, render_demo, returns, summary)


OPENING = OUTPUT / 'rally-opening-demonstrations.npz'
RECEIPT = OUTPUT / 'rally-from-release.json.gz'
FRESH = OUTPUT / 'rally-from-release-fresh.json.gz'
SETTINGS = dict(neighbors=20, body=.68, mix=0.)
DIMS = (480, 360)
TRIAL = None


def collect(path, workers):
    held = archive_schedule()
    schedule = demo_schedule(held)
    render_prefix = partial(render_demo, stop_at=HANDOVER, first_sample=0)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        blocks = list(pool.map(render_prefix, schedule))
    examples = {key: np.concatenate([block[key] for block in blocks]) for key in ('features', 'postures', 'rates', 'seed')}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **examples, demo_seed=[row['seed'] for row in schedule],
                        opening_frames=[block['frame_count'] for block in blocks])


def prepare(opening, continuation, profile, dims):
    global TRIAL
    with np.load(opening) as left, np.load(continuation) as right:
        columns = [np.concatenate((left[key], right[key])) for key in ('features', 'postures', 'rates')]
    policy = JointPolicy(*columns, STEP_SIZE, **SETTINGS)
    TRIAL = Station(profile, render_dims=dims), policy


def test_one(index, row):
    station, policy = TRIAL
    order = ('hold', 'sight_delay', 'live') if index % 2 == 0 else ('live', 'sight_delay', 'hold')
    start = dict(seed=row['seed'], lane=row['lane'])
    reports = {name: station.play(start, name, policy, live_opening=True) for name in order}
    if len({report['prefix_rgb'] for report in reports.values()}) != 1:
        raise ValueError('initial release RGB differs among treatments')
    for report in reports.values():
        report['vs_hold'] = difference_count(report, reports['hold'], 0)
    return index, dict(start, results=reports)


def run(path, opening, data, workers, *, fresh=False, dims=DIMS):
    held = course(REPLICATION) if fresh else archive_schedule()
    profile = _read(COACHED)['metadata']['opening']['far_profile']
    with np.load(opening) as one, np.load(data) as two:
        seen = set(one['seed']) | set(two['seed'])
        settings = dict(schema='tt24-rgb-feedback-from-release-1', teacher_only_during_training=True,
                        teacher_commands_during_test=0, opening_data_sha256=_hash(opening),
                        continuation_data_sha256=_hash(data), opening_samples=len(one['seed']),
                        continuation_samples=len(two['seed']), settings=SETTINGS,
                        fit_seeds=sorted(map(int, seen)), render_dims=list(dims))
    if fresh:
        settings['fresh_schedule'] = REPLICATION
    if seen.intersection(row['seed'] for row in held):
        raise ValueError('held release overlaps demonstrations')
    result = dict(metadata=settings, cases=[None]*len(held))
    if path.exists():
        result = _read(path)
        if result['metadata'] != settings:
            raise ValueError('from-release checkpoint uses different input data')
    path.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=prepare,
                             initargs=(opening, data, profile, dims)) as pool:
        tasks = [pool.submit(test_one, i, row) for i, row in enumerate(held) if result['cases'][i] is None]
        for task in as_completed(tasks):
            index, row = task.result()
            result['cases'][index] = row
            _write(path, result)
            print(row['seed'], {name: returns(branch, 1) for name, branch in row['results'].items()}, flush=True)
    result['verified_tapes'] = verify(result['cases'])
    result['summary'] = {}
    for name in ('hold', 'sight_delay', 'live'):
        counts = summary(result['cases'], name, 0)
        counts['reach_first'] = sum(n >= 1 for n in counts['own_returns'])
        result['summary'][name] = counts
    _write(path, result)
    return result['summary']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--opening-data', type=Path, default=OPENING)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--size', type=int, choices=(240, 320, 480), default=480)
    args = parser.parse_args()
    if not args.opening_data.exists():
        collect(args.opening_data, args.workers)
    output = args.output or (FRESH if args.fresh else RECEIPT)
    print(run(output, args.opening_data, args.data, args.workers, fresh=args.fresh,
              dims=(args.size, args.size*3//4)), flush=True)
