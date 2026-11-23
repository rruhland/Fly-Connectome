"""Repeat the broader fit with the original policy's local smoothing."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt23_sequence import verify
from .tt24_rally_bank import BASE_SETTINGS, BROAD_SETTINGS, CACHE, DIMS, RECEIPT as MAIN
from .tt24_rally_from_release import OPENING
from .tt24_rally_policy import JointPolicy
from .tt24_rally_trial import COACHED, DATA, OUTPUT, STEP_SIZE, Station, difference_count, returns, summary


RECEIPT = OUTPUT / 'rally-bank-matched-locality.json.gz'
ORIGINAL = OUTPUT / 'rally-bank-original-wide-locality.json.gz'


def run(path, *, original=False):
    baseline = _read(MAIN)
    if baseline['verified_tapes'] != sum(len(row['results']) for row in baseline['cases']):
        raise ValueError('complete the main E/F comparison before running its matched-locality check')
    keys = ('features', 'postures', 'rates')
    if original:
        with np.load(OPENING) as first, np.load(DATA) as next_:
            values = [np.concatenate((first[key], next_[key])) for key in keys]
        trial = dict(training_sha256=[_hash(OPENING), _hash(DATA)], settings=BROAD_SETTINGS)
    else:
        with np.load(CACHE) as array:
            values = [array[key] for key in keys]
        trial = dict(training_sha256=_hash(CACHE), settings=BASE_SETTINGS)
    policy = JointPolicy(*values, STEP_SIZE, **trial['settings'])
    profile = _read(COACHED)['metadata']['opening']['far_profile']
    station = Station(profile, render_dims=DIMS)
    meta = dict(assessment_sha256=_hash(MAIN), extra_teacher_frames=0, held_teacher_commands=0, **trial)
    report = _read(path) if path.exists() else dict(metadata=meta, cases=[])
    if report['metadata'] != meta:
        raise ValueError('the held course or fit changed')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for case in baseline['cases'][len(report['cases']):]:
            start = {name: case[name] for name in ('seed', 'lane', 'cohort', 'stratum')}
            row = station.play(start, 'matched_locality', policy, live_opening=True)
            reference = case['results']['original']
            if reference['prefix_rgb'] != row['prefix_rgb']:
                raise ValueError('common initial observation differs')
            row['vs_original'] = difference_count(row, reference, 0)
            report['cases'].append(dict(start, results={'matched_locality': row}))
            _write(path, report)
            print(start['seed'], returns(row, 1), flush=True)
    finally:
        station.arena.close()
    report['verified_tapes'] = verify(report['cases'])
    report['summary'] = checked = summary(report['cases'], 'matched_locality', 0)
    checked['reach_first'] = sum(n > 0 for n in checked['own_returns'])
    _write(path, report)
    return checked


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--original', action='store_true')
    args = parser.parse_args()
    print(run(args.output or (ORIGINAL if args.original else RECEIPT), original=args.original), flush=True)
