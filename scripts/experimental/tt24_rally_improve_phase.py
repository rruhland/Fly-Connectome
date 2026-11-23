"""Blend two direct joint feedback fits after a camera-observed reversal."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt23_sequence import REPLICATION, course, verify
from .tt24_rally_bank import CACHE, DIMS, HOLDOUT
from .tt24_rally_improve import stereo_motion
from .tt24_rally_improve_trial import CONFIRMATION, policy_for, quality
from .tt24_rally_policy import ViewHistory
from .tt24_rally_trial import COACHED, OUTPUT, Station, difference_count, returns, summary


DEV = OUTPUT / 'rally-improve-phase-development.json.gz'
HELD = OUTPUT / 'rally-improve-phase-held.json.gz'
FRESH = {'m': 1976954423, 'n': 640805439}
CANDIDATES = ('rate_blend', 'relative_blend')
TRIAL = None


class PhaseHistory(ViewHistory):
    """Retain visually observed reversal and optionally delay visual feedback."""

    def __init__(self, *, delayed=False):
        super().__init__()
        self.approached = False
        self.finished = False
        self.amount = 0.
        self.delayed = delayed
        self.past = []

    def observe(self, frames, joints, issued_position):
        state = super().observe(frames, joints, issued_position)
        if self.delayed:
            self.past.append(state[:14].copy())
            state[:14] = self.past[max(0, len(self.past)-7)]
        motion = stereo_motion(state)
        y, vy = motion[[1, 4]]
        quality_ok = np.max(state[[6, 13]]) <= 1
        if quality_ok and y < -.50 and vy < -.024:
            self.approached = True
        if self.approached and quality_ok and y < -.40 and vy > .028:
            self.finished = True
        if self.finished:
            self.amount = min(1., self.amount + .31)
        return np.r_[state, self.amount]


class PhasePolicy:
    """Smoothly change the weights of continuous feedback readouts."""

    def __init__(self, original, further, fraction):
        self.original = original
        self.further = further
        self.fraction = fraction

    def command(self, observation, issued):
        state, gate = observation[:-1], observation[-1]
        opening = self.original.command(state, issued)
        if gate <= 0:
            return opening
        continuation = self.further.command(state, issued)
        fraction = self.fraction*gate
        return opening + fraction*(continuation-opening)


def prepare(profile, selected):
    global TRIAL
    with np.load(CACHE) as data:
        broad = policy_for('broad', data)
        models = dict(broad=broad,
                      rate_blend=PhasePolicy(broad, policy_for('retina_rate', data), .82),
                      relative_blend=PhasePolicy(broad, policy_for('relative_contrast', data), .78))
    TRIAL = Station(profile, render_dims=DIMS), models, selected


def one_start(start, order):
    station, models, selected = TRIAL
    result = {}
    for name in order:
        if name in ('hold', 'broad'):
            row = station.play(start, name, models.get(name), live_opening=True)
        else:
            policy = models[selected if name == 'phase_delay' else name]
            row = station.play(start, name, policy, lambda: PhaseHistory(delayed=name == 'phase_delay'),
                               live_opening=True)
        result[name] = row
    if len({row['prefix_rgb'] for row in result.values()}) != 1:
        raise ValueError('the pupils did not start with the same view')
    for row in result.values():
        row['vs_broad'] = difference_count(row, result['broad'], 0)
    return dict(start, results=result)


def evaluate(path, workers, confirm):
    if confirm:
        dev = _read(DEV)
        if not dev.get('verified_tapes'):
            raise ValueError('finish and verify development first')
        selected = max(CANDIDATES, key=lambda name: quality(dev['summary'][name]))
        starts = course(FRESH)
        modes = ('broad', selected, 'phase_delay', 'hold')
    else:
        selected = None
        starts = course(HOLDOUT)[:12]
        modes = ('broad', *CANDIDATES)
    used = {row['seed'] for row in course(HOLDOUT)+course(REPLICATION)+course(CONFIRMATION)}
    with np.load(CACHE) as data:
        known = set(map(int, data['seed']))
    if confirm and (known | used) & {row['seed'] for row in starts}:
        raise ValueError('confirmation overlaps exposure')
    profile = _read(COACHED)['metadata']['opening']['far_profile']
    meta = dict(schema='tt24-camera-reversal-blend-1', starts=starts, modes=list(modes), selected=selected,
                teacher_commands=0, demo_sha256=_hash(CACHE), dimensions=list(DIMS))
    output = _read(path) if path.exists() else dict(metadata=meta, cases=[None]*len(starts))
    if output['metadata'] != meta:
        raise ValueError('test metadata differ')
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=prepare,
                             initargs=(profile, selected)) as pool:
        tasks = {}
        for index, start in enumerate(starts):
            if output['cases'][index] is None:
                offset = index % len(modes)
                order = modes[offset:] + modes[:offset]
                tasks[pool.submit(one_start, start, order)] = index
        for task in as_completed(tasks):
            case = task.result()
            output['cases'][tasks[task]] = case
            _write(path, output)
            print('phase', case['seed'], {key: returns(row, 1) for key,row in case['results'].items()}, flush=True)
    output['verified_tapes'] = verify(output['cases'])
    output['summary'] = {mode: summary(output['cases'], mode, 0) for mode in modes}
    for row in output['summary'].values():
        row['reach_first'] = sum(n > 0 for n in row['own_returns'])
    _write(path, output)
    return output['summary']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--confirm', action='store_true')
    args = parser.parse_args()
    print(evaluate(args.output or (HELD if args.confirm else DEV), args.workers, args.confirm), flush=True)
