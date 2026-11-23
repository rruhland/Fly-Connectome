"""Score new direct feedback architectures on development, then on unrelated releases."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
from multiprocessing import get_context
from pathlib import Path
import re

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt23_sequence import REPLICATION, course, verify
from .tt24_rally_bank import ARCHIVE, BROAD_SETTINGS, CACHE, DIMS, HOLDOUT
from .tt24_rally_improve import MotionPolicy
from .tt24_rally_policy import JointPolicy
from .tt24_rally_trial import (COACHED, OUTPUT, STEP_SIZE, Station, archive_schedule,
                               difference_count, returns, summary)


DEVELOPMENT = OUTPUT / 'rally-improve-development.json.gz'
HELD = OUTPUT / 'rally-improve-held.json.gz'
SCHEDULES = OUTPUT / 'rally-improve-schedules.json.gz'
MODES = ('stereo', 'relative', 'retina_rate', 'relative_contrast', 'moving')
CONFIRMATION = {'g': 1749010825, 'h': 2106926783}
TRIAL = None


def policy_for(name, data):
    columns = [data[key] for key in ('features', 'postures', 'rates')]
    if name == 'broad':
        return JointPolicy(*columns, STEP_SIZE, **BROAD_SETTINGS)
    return MotionPolicy(*columns, STEP_SIZE, mode=name)


def prepare(modes, profile, selected):
    global TRIAL
    with np.load(CACHE) as data:
        policies = {name: policy_for(name, data) for name in modes if name in ('broad', *MODES)}
    if 'sight_delay' in modes:
        policies['sight_delay'] = policies[selected]
    TRIAL = Station(profile, render_dims=DIMS), policies


def one_start(order, start):
    station, policies = TRIAL
    result = {name: station.play(start, name, policies.get(name), live_opening=True) for name in order}
    if len({row['prefix_rgb'] for row in result.values()}) != 1:
        raise ValueError('the two pupils saw different initial RGB')
    if 'broad' in result:
        for row in result.values():
            row['vs_broad'] = difference_count(row, result['broad'], 0)
    return dict(start, results=result)


def choose(development):
    data = _read(development)
    if data['verified_tapes'] != sum(len(row['results']) for row in data['cases']):
        raise ValueError('verify the development comparisons before freezing the candidate')
    return max(data['metadata']['conditions'], key=lambda name: quality(data['summary'][name]))


def quality(outcome):
    return (outcome['reach_fifth'], outcome['reach_second'], outcome['total_own_returns'], outcome['reach_first'])


def evaluate(path, modes, starts, workers, *, selection=None):
    profile = _read(COACHED)['metadata']['opening']['far_profile']
    meta = dict(schema='tt24-live-visual-motion-relative-command-1', conditions=list(modes),
                starts=starts, demos_sha256=_hash(CACHE), profile=profile, dimensions=list(DIMS),
                selected=selection, teacher_commands=0, teacher_bank='48 archived separate releases')
    with np.load(CACHE) as bank:
        if set(map(int, bank['seed'])) & {row['seed'] for row in starts}:
            raise ValueError('an assessment launch leaked into teacher data')
    output = _read(path) if path.exists() else dict(metadata=meta, cases=[None]*len(starts))
    if output['metadata'] != meta:
        raise ValueError('assessment changed')
    path.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=prepare,
                             initargs=(modes, profile, selection)) as pool:
        tasks = {}
        for i, start in enumerate(starts):
            if output['cases'][i] is None:
                offset = i % len(modes)
                order = modes[offset:] + modes[:offset]
                tasks[pool.submit(one_start, order, start)] = i
        for task in as_completed(tasks):
            result = task.result()
            output['cases'][tasks[task]] = result
            print(result['seed'], {name: returns(row, 1) for name, row in result['results'].items()}, flush=True)
            _write(path, output)
    output['verified_tapes'] = verify(output['cases'])
    output['summary'] = {}
    for name in modes:
        values = summary(output['cases'], name, 0)
        values['reach_first'] = sum(count >= 1 for count in values['own_returns'])
        output['summary'][name] = values
    _write(path, output)
    return output['summary']


def new_starts():
    starts = course(CONFIRMATION)
    known = {row['seed'] for row in course(HOLDOUT) + course(REPLICATION)}
    if known & {row['seed'] for row in starts}:
        raise ValueError('confirmation releases overlap prior held cohorts')
    return starts


def audit_schedules(path):
    from .tt24_rally_improve_optical import FRESH as OPTICAL
    from .tt24_rally_improve_phase import FRESH as PHASE
    from .tt24_rally_improve_practice import CONTINUATION_FRESH, FRESH as SCALAR
    from .tt24_rally_improve_recovery import FRESH as RECOVERY

    roster = {name: course(generators) for name, generators in
              dict(rate=CONFIRMATION, scalar=SCALAR, extent=OPTICAL, phase=PHASE,
                   continuation=CONTINUATION_FRESH, recovery=RECOVERY).items()}
    reserved = [row['seed'] for starts in roster.values() for row in starts]
    known = {row['seed'] for row in course(HOLDOUT) + course(REPLICATION) + archive_schedule()}
    with np.load(CACHE) as data:
        known.update(map(int, data['seed']))
    if len(set(reserved)) != len(reserved) or known.intersection(reserved):
        raise ValueError('a reserved launch appears in development or demonstrations')
    literals = b'|'.join(str(seed).encode() for seed in sorted(reserved))
    pattern = re.compile(rb'(?<![0-9.])(?:' + literals + rb')(?![0-9.])')
    scanned = 0
    for directory in (ARCHIVE.with_name('tt22'), ARCHIVE):
        for receipt in sorted(directory.iterdir()):
            if receipt.name.endswith('.json.gz'):
                raw = gzip.decompress(receipt.read_bytes())
            elif receipt.suffix == '.json':
                raw = receipt.read_bytes()
            else:
                continue
            if pattern.search(raw):
                raise ValueError(f'a reserved launch already appears in {receipt}')
            scanned += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, dict(schedules=roster, seeds=len(reserved), archived_files_checked=scanned))
    return dict(reserved=len(reserved), archived_files_checked=scanned)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--confirm', action='store_true')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--only', nargs='+', choices=MODES, default=MODES)
    parser.add_argument('--limit', type=int, default=12)
    args = parser.parse_args()
    if args.audit:
        result = audit_schedules(args.output or SCHEDULES)
    elif args.confirm:
        candidate = choose(DEVELOPMENT)
        result = evaluate(args.output or HELD, ('broad', candidate, 'sight_delay', 'hold'),
                          new_starts(), args.workers, selection=candidate)
    else:
        result = evaluate(args.output or DEVELOPMENT, tuple(args.only), course(HOLDOUT)[:args.limit], args.workers)
    print(result, flush=True)
