"""Refine the demonstrated framewise rally policy with own delayed point outcomes."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt23_sequence import HANDOVER, alternating, verify
from .tt24_rally_trial import (COACHED, DATA, OUTPUT, Station, archive_schedule, archived_rows,
                               contacts, difference_count, make_policy, returns, summary)


RECEIPT = OUTPUT / 'rally-reward-refinement.json.gz'
PRACTICE_SEED = 179271280
SEARCH_SEED = 850338
DIMS = (240, 180)
CONDITIONS = ('hold', 'warm', 'refined', 'sight_delay')


class ResidualPolicy:
    """A bounded perturbation of the current observation-to-rate function."""

    def __init__(self, base, weights):
        self.base = base
        self.weights = weights

    def command(self, state, issued_position):
        command = self.base.command(state, issued_position)
        left, right = state[:7], state[7:14]
        lateral = np.tanh((left[0] + right[0] - 480.) / 38.)
        depth_motion = np.tanh((left[2] - right[2]) / 6.)
        basis = np.array((1., lateral, depth_motion))
        delta = np.clip(self.weights @ basis, -.55, .55)
        command[:5] = np.clip(command[:5] + delta, -1., 1.)
        return command


def score(row):
    legal = sum(tick > HANDOVER for tick in returns(row, 1)[1:])
    touches = sum(tick > HANDOVER for tick in contacts(row, 1))
    length = min(15, alternating(row['events']))
    return legal + .09*touches + .02*length


def train(station, base, practice, budget, report, path):
    rng = np.random.default_rng(SEARCH_SEED)
    anchor = np.zeros((5, 3))
    best = (-np.inf, anchor.copy())
    practice_rows = report['practice']
    for trial in range(budget):
        if trial == 0:
            candidate = anchor.copy()
            perturbation = np.zeros_like(candidate)
        elif trial % 2:
            parent = anchor.copy()
            step = rng.choice((.07, .14, .22), p=(.27, .53, .20))
            perturbation = rng.normal(size=(5, 1)) * rng.normal(size=(1, 3)) * step
            candidate = np.clip(parent + perturbation, -.58, .58)
        else:
            candidate = np.clip(parent - perturbation, -.58, .58)
        if trial < len(practice_rows):
            entry = practice_rows[trial]
            if not np.array_equal(candidate, entry['weights']):
                raise ValueError('reward search diverged from checkpoint')
            row = entry['result']
        else:
            row = station.play(practice, 'practice', ResidualPolicy(base, candidate))
            practice_rows.append(dict(weights=candidate.tolist(), score=score(row), result=row))
            _write(path, report)
            print('reward', trial + 1, returns(row, 1), row['outcome']['reason'], round(score(row), 3), flush=True)
        quality = score(row)
        if quality > best[0]:
            best = quality, candidate.copy()
        # End the antithetic pair before changing its center.
        if trial % 2 == 0:
            anchor = best[1].copy()
    return best


def run(path, data_path, budget):
    starts = archive_schedule()
    practice = max((row for row in archived_rows(COACHED) if row['seed'] == PRACTICE_SEED),
                   key=lambda row: len(returns(row, 1)))
    if PRACTICE_SEED in {row['seed'] for row in starts}:
        raise ValueError('reward practice overlaps held releases')
    profile = _read(COACHED)['metadata']['opening']['far_profile']
    with np.load(data_path) as data:
        base, exposure = make_policy(data, 'demo_full')
    station = Station(profile, render_dims=DIMS)
    meta = dict(schema='tt24-joint-reward-refinement-1', training_seed=PRACTICE_SEED, search_seed=SEARCH_SEED,
                schedule=[row['seed'] for row in starts], teacher=exposure, data_sha256=_hash(data_path),
                dimensions=list(DIMS), budget=budget, conditions=list(CONDITIONS))
    report = _read(path) if path.exists() else dict(metadata=meta, practice=[], cases=[None]*len(starts))
    if report['metadata'] != meta:
        raise ValueError('reward refinement checkpoint inputs changed')
    try:
        best_score, best_weights = train(station, base, practice, budget, report, path)
        report['best'] = dict(score=best_score, weights=best_weights.tolist())
        learned = ResidualPolicy(base, best_weights)
        frozen = ResidualPolicy(base, np.zeros((5, 3)))
        for index, start in enumerate(starts):
            if report['cases'][index] is not None:
                continue
            order = CONDITIONS if index % 2 == 0 else CONDITIONS[::-1]
            rows = {}
            for name in order:
                rows[name] = station.play(start, name, frozen if name in ('hold', 'warm') else learned)
            if len({row['prefix_rgb'] for row in rows.values()}) != 1:
                raise ValueError('shared teacher opening rendered differently')
            for row in rows.values():
                row['vs_hold'] = difference_count(row, rows['hold'])
            report['cases'][index] = dict(seed=start['seed'], teacher_prefix=start['actions'][:HANDOVER], results=rows)
            _write(path, report)
            print('held residual', start['seed'], {name: returns(row, 1) for name, row in rows.items()}, flush=True)
    finally:
        station.arena.close()
    replay = report['cases'] + [dict(results={'practice': visit['result']}) for visit in report['practice']]
    report['verified_tapes'] = verify(replay)
    report['summary'] = {name: summary(report['cases'], name) for name in CONDITIONS}
    report['target_practice_frames'] = sum(len(entry['result']['actions'])-HANDOVER for entry in report['practice'])
    _write(path, report)
    return report['summary']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RECEIPT)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--budget', type=int, default=19)
    args = parser.parse_args()
    print(run(args.output, args.data, args.budget), flush=True)
