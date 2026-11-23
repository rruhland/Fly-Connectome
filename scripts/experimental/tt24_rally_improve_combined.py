"""Refit optical feedback on pupil-induced recoveries and score an independent cohort."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
from multiprocessing import get_context
from pathlib import Path
import re

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt23_sequence import REPLICATION, course, verify
from .tt24_rally_bank import ARCHIVE, DIMS, HOLDOUT
from .tt24_rally_improve_optical import (CACHE, ExtentHistory, OpticalPolicy, collect_tape,
                                       prepare_collection)
from .tt24_rally_improve_phase import PhasePolicy
from .tt24_rally_improve_recovery import PRACTICE
from .tt24_rally_improve_trial import SCHEDULES
from .tt24_rally_trial import COACHED, OUTPUT, STEP_SIZE, Station, difference_count, returns, summary


DATA = OUTPUT / 'rally-improve-combined-demos.npz'
HELD = OUTPUT / 'rally-improve-combined-held.json.gz'
FRESH = {'s': 997342783, 't': 1026558929}
MODES = ('optical', 'combined', 'combined_delay', 'hold')
WORKER = None


class PhaseExtent(ExtentHistory):
    def __init__(self, sphere, *, delayed=False):
        super().__init__(sphere, delayed=delayed)
        self.approached = False
        self.reversed = False
        self.amount = 0.

    def observe(self, frames, joints, position):
        state = super().observe(frames, joints, position)
        y, speed, wait = state[[39, 42, 49]]
        if wait <= 1 and y < -.50 and speed < -.024:
            self.approached = True
        if self.approached and wait <= 1 and y < -.40 and speed > .028:
            self.reversed = True
        if self.reversed:
            self.amount = min(1., self.amount + .31)
        return np.r_[state, self.amount]


def reconstruct(path, workers):
    practice = _read(PRACTICE)
    if practice['verified_tapes'] != len(practice['construction']):
        raise ValueError('the corrective tapes must pass verification first')
    tutors = [dict(row, refit_recovery=True) for row in practice['construction'] if row['fit_frames']]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                             initializer=prepare_collection) as pool:
        blocks = list(pool.map(collect_tape, [(None, row) for row in tutors]))
    with np.load(CACHE) as original:
        sphere = original['sphere']
        data = {key: original[key].tolist() for key in ('features', 'postures', 'rates', 'seed')}
    for tutor, block in zip(tutors, blocks, strict=True):
        reader = ExtentHistory(sphere)
        for _, include, observation, joints, issued, rate in block:
            state = reader.read(observation, joints, issued)
            reader.issued(rate)
            if include:
                data['features'].append(state)
                data['postures'].append(issued + STEP_SIZE*rate)
                data['rates'].append(rate)
                data['seed'].append(tutor['seed'])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data, sphere=sphere, extra_fit=practice['cost']['fit_frames'],
                        teacher_frames=practice['cost']['teacher_frames'], corrective_replays=len(tutors),
                        source_sha256=_hash(PRACTICE), optical_sha256=_hash(CACHE))


def prepare(path, profile):
    global WORKER
    with np.load(path) as data, np.load(CACHE) as original:
        optical = OpticalPolicy(original, 'interaction')
        augmented = OpticalPolicy(data, 'interaction')
        sphere = data['sphere']
    stage = PhasePolicy(optical, augmented, .88)
    WORKER = Station(profile, render_dims=DIMS), optical, stage, sphere


def trial(start, modes):
    station, optical, combined, sphere = WORKER
    outcomes = {}
    for name in modes:
        if name in ('optical', 'hold'):
            row = station.play(start, name, optical, lambda: ExtentHistory(sphere), live_opening=True)
        else:
            row = station.play(start, name, combined, lambda: PhaseExtent(sphere, delayed=name=='combined_delay'),
                               live_opening=True)
        outcomes[name] = row
    if len({row['prefix_rgb'] for row in outcomes.values()}) != 1:
        raise ValueError('initial RGB changed across the trials')
    for row in outcomes.values():
        row['vs_optical'] = difference_count(row, outcomes['optical'], 0)
    return dict(start, results=outcomes)


def fresh(path):
    starts = course(FRESH)
    proposed = {row['seed'] for row in starts}
    known = {row['seed'] for row in course(REPLICATION)+course(HOLDOUT)}
    known.update(row['seed'] for schedule in _read(SCHEDULES)['schedules'].values() for row in schedule)
    with np.load(path) as data:
        known.update(map(int, data['seed']))
    if len(proposed) != len(starts) or known.intersection(proposed):
        raise ValueError('held release overlaps a prior experiment')
    pattern = re.compile(rb'(?<![0-9.])(?:'+b'|'.join(str(seed).encode() for seed in sorted(proposed))+rb')(?![0-9.])')
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
                raise ValueError(f'held release overlaps {receipt}')
            scanned += 1
    return starts, scanned


def assess(output, path, workers):
    starts, past_archives = fresh(path)
    profile = _read(COACHED)['metadata']['opening']['far_profile']
    meta = dict(schema='tt24-optical-and-corrective-feedback-1', archive_files_checked=past_archives,
                teacher_commands_in_assessment=0, dimensions=list(DIMS), modes=list(MODES),
                withheld_starts=starts, original_optical_sha256=_hash(CACHE), corrective_sha256=_hash(path))
    data = _read(output) if output.exists() else dict(metadata=meta, cases=[None]*len(starts))
    if data['metadata'] != meta:
        raise ValueError('test inputs changed')
    output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=prepare,
                             initargs=(path, profile)) as pool:
        pending = {}
        for i, start in enumerate(starts):
            if data['cases'][i] is None:
                p = i % len(MODES)
                pending[pool.submit(trial, start, MODES[p:]+MODES[:p])] = i
        for task in as_completed(pending):
            row = task.result()
            data['cases'][pending[task]] = row
            _write(output, data)
            print('combined', row['seed'], {mode: returns(result, 1) for mode,result in row['results'].items()}, flush=True)
    data['verified_tapes'] = verify(data['cases'])
    data['summary'] = {mode: summary(data['cases'], mode, 0) for mode in MODES}
    for outcome in data['summary'].values():
        outcome['reach_first'] = sum(v>0 for v in outcome['own_returns'])
    _write(output, data)
    return data['summary']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=HELD)
    parser.add_argument('--workers', type=int, default=1)
    settings = parser.parse_args()
    if not settings.dataset.exists():
        reconstruct(settings.dataset, settings.workers)
    print(assess(settings.output, settings.dataset, settings.workers), flush=True)
