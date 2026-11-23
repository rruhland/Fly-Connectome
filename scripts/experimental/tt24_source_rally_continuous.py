"""Apply a continuous local update to table joint feedback after matched RGB practice."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt23_sequence import course, verify
from .tt24_rally_fresh import SETTINGS
from .tt24_rally_trial import COACHED, DATA, OUTPUT, Station, difference_count, make_policy, returns, summary
from .tt24_source import CHECKPOINT, MODES, VisualConsequence
from .tt24_source_rally import RECEIPT as PRACTICE
from .tt24_source_rally import chosen, fresh_openings, score


RECEIPT = OUTPUT / 'source-rally-continuous.json.gz'
COHORTS = {'continuous_a': 947888972, 'continuous_b': 1105217770}
RIDGE = .30
UPDATE_SPAN = .28


def updates(practice):
    draw = np.array(practice['metadata']['emission'])
    picks, values = chosen(practice['practice'])
    anchor = draw[picks['cold']]
    halfway = (len(draw) + 1) // 2
    directions = draw[1:halfway].reshape(halfway - 1, -1)
    penalty = directions @ directions.T + RIDGE * np.eye(halfway - 1)
    gradients = {}
    for mode, value in values.items():
        utility = np.asarray(value)
        differences = (utility[1:halfway] - utility[halfway:]) / 2.
        gradients[mode] = (directions.T @ np.linalg.solve(penalty, differences)).reshape(anchor.shape)
    # Set a shared step from the public-reward fit. Keep the magnitude in the
    # same coordinate range as the joint readouts physically tried in practice.
    normalizer = max(.04, np.linalg.norm(gradients['cold']))
    readouts = {mode: np.clip(anchor + UPDATE_SPAN * d / normalizer, -1., 1.) for mode, d in gradients.items()}
    return picks['cold'], gradients, readouts


def run(path, practice_path, data_path, count):
    began = perf_counter()
    trial = _read(practice_path)
    native_hash = _hash(CHECKPOINT)
    if trial['metadata']['source_sha256'] != native_hash or _hash(data_path) != trial['metadata']['demo_data_sha256']:
        raise ValueError('continuous practice needs the recorded demonstration and frozen Pong weights')
    position, gradients, weights = updates(trial)
    starts = course(COHORTS)
    known = set(trial['metadata']['practice_starts']) | set(trial['metadata']['archived_held_starts'])
    known |= {row['seed'] for row in trial['metadata']['fresh_schedule']}
    if {row['seed'] for row in starts} & known:
        raise ValueError('continuous assessment overlaps an earlier source or target course')
    opening = _read(COACHED)['metadata']['opening']
    # Hash only immutable practice rows. The original receipt can acquire its
    # own evaluation results while this disjoint physical check is running.
    practice_fingerprint = trial['metadata'] | dict(practice_scores=chosen(trial['practice'])[1])
    metadata = dict(schema='tt24-visual-credit-continuous-1', target_practice=practice_fingerprint,
                    fixed_schedule=starts, count=count, common_anchor=position,
                    gradients={name: a.tolist() for name, a in gradients.items()},
                    learned={name: a.tolist() for name, a in weights.items()}, ridge=RIDGE,
                    update_span=UPDATE_SPAN, source_sha256=native_hash)
    report = _read(path) if path.exists() else dict(metadata=metadata)
    if report['metadata'] != metadata:
        raise ValueError('earlier continuous receipt has different practice or target weights')
    if 'opening_rows' not in report:
        screen = Station(opening['far_profile'])
        try:
            report['opening_rows'], report['screen'] = fresh_openings(opening, screen, starts, count)
        finally:
            screen.arena.close()
        report['cases'] = [dict(seed=row['seed'], results={}) for row in report['opening_rows']]
        _write(path, report)
    models = {mode: VisualConsequence.load(mode) for mode in MODES}
    station = Station(opening['far_profile'], render_dims=tuple(trial['metadata']['render_dimensions']))
    with np.load(data_path) as data:
        base, _ = make_policy(data, 'demo_full', **SETTINGS)
    try:
        for i, (start, row) in enumerate(zip(report['opening_rows'], report['cases'], strict=True)):
            for mode in MODES if i % 2 == 0 else MODES[::-1]:
                if mode in row['results']:
                    continue
                played = score(station, start, base, weights[mode], models)
                row['results'][mode] = played
                _write(path, report)
                print('continuous', start['seed'], mode, returns(played, 1), len(played['actions']),
                      round(perf_counter() - began, 1), flush=True)
            if len({r['prefix_rgb'] for r in row['results'].values()}) != 1:
                raise ValueError('continuous treatments had different shared RGB openings')
            for mode in MODES:
                row['results'][mode]['vs_cold'] = difference_count(row['results'][mode], row['results']['cold'])
        report['verified_tapes'] = verify(report['cases'])
        report['summary'] = {mode: summary(report['cases'], mode) for mode in MODES}
        report['seconds'] = perf_counter() - began
        _write(path, report)
        print(report['summary'], flush=True)
    finally:
        station.arena.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RECEIPT)
    parser.add_argument('--practice', type=Path, default=PRACTICE)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--count', type=int, default=4)
    args = parser.parse_args()
    run(args.output, args.practice, args.data, args.count)


if __name__ == '__main__':
    main()
