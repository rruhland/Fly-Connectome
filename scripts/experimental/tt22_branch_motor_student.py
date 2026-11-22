"""Add strict counterfactual outcomes at response cues visited by the frozen student."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_branch_motor import BranchMotor, LATERAL_FRACTION, MOTORS, OutcomeScorer
from .tt22_branch_motor_trial import BranchPractice, NativePoint, OUTPUT
from .tt22_opponent_learn import _replay

PRACTICE = OUTPUT/'branch-motor-practice.json.gz'
STUDENT_TRACE = OUTPUT/'branch-motor-student-native.json.gz'
CHECKPOINT = OUTPUT/'branch-motor-student-practice.json.gz'
BASELINE = OUTPUT/'branch-motor-student-baseline.json.gz'
RECEIPT = OUTPUT/'branch-motor-student-held.json.gz'
TRAIN_SCHEDULE = 402687021
HELD_SCHEDULE = 1731922005
TRAIN_COUNT = 12
HELD_COUNT = 8


def record(checkpoint, output, count, schedule, *, scorer=OutcomeScorer, policy='contextual'):
    weights = _read(checkpoint)
    if weights['motors'] != [list(motor) for motor in MOTORS]:
        raise ValueError('practice motor library and live library disagree')
    seeds = np.random.default_rng(schedule).integers(2**31, size=count).tolist()
    if len(set(seeds)) != count or set(seeds) & {row[1] for row in weights['donors']}:
        raise ValueError('native schedule includes duplicate or practice seeds')
    coefficients = weights['source']
    metadata = dict(schema='tt22-branch-held-1', weights=checkpoint.name, weights_hash=_hash(checkpoint),
                    seeds=seeds, schedule=schedule, coefficients={policy: coefficients},
                    profile=weights['profile'], lateral_fraction=LATERAL_FRACTION)
    run = dict(metadata, rows={policy: []})
    if output.exists():
        run = _read(output)
        if any(run.get(key) != value for key, value in metadata.items()):
            raise ValueError(f'cannot resume {output.name} with changed weights or schedule')
    rows = run['rows'][policy]
    if [row['seed'] for row in rows] != seeds[:len(rows)]:
        raise ValueError(f'{output.name} does not contain a prefix of the expected seeds')
    if run.get('verified') == count and len(rows) == count:
        return
    started = perf_counter()
    if len(rows) < count:
        learner = BranchMotor(coefficients, scorer(weights['contexts']))
        point = NativePoint(weights)
        try:
            for seed in seeds[len(rows):]:
                row = point.play(seed, learner)
                rows.append(row)
                _write(output, run)
                print(policy, seed, row['longest'], row['outcome']['legal_returns'], row['outcome']['reason'],
                      round(perf_counter()-started, 1), flush=True)
        finally:
            point.close()
    run['verified'] = _replay(rows)
    run['seconds'] = run.get('seconds', 0.) + perf_counter()-started
    _write(output, run)


def aggregate(base_path, trace_path, checkpoint):
    base, trace = _read(base_path), _read(trace_path)
    rows = trace['rows']['contextual']
    seeds = np.random.default_rng(TRAIN_SCHEDULE).integers(2**31, size=TRAIN_COUNT).tolist()
    if (trace['weights_hash'] != _hash(base_path) or trace['schedule'] != TRAIN_SCHEDULE
            or [row['seed'] for row in rows] != seeds or trace.get('verified') != TRAIN_COUNT):
        raise ValueError('student archive must be complete and replay-verified under the frozen teacher-trained policy')
    training = {row['seed'] for row in rows}
    if len(training) != len(rows) or training & {row[1] for row in base['donors']}:
        raise ValueError('student and original donor practice starts must be disjoint')
    checkpoint_data = dict(base, schema='tt22-branch-practice-student-1', original_hash=_hash(base_path),
                           original_contexts=len(base['contexts']),
                           student_source=dict(file=trace_path.name, sha256=_hash(trace_path),
                                               schedule=TRAIN_SCHEDULE, points=len(rows),
                                               native_frames=sum(len(row['actions']) for row in rows)))
    original_donors = len(base['donors'])
    if checkpoint.exists():
        previous = _read(checkpoint)
        lineage = ('schema', 'original_hash', 'original_contexts', 'student_source')
        if any(previous[key] != checkpoint_data[key] for key in lineage):
            raise ValueError('cannot resume student practice with changed archives')
        checkpoint_data = previous
    added_donors = [row[1] for row in checkpoint_data['donors'][original_donors:]]
    if added_donors != [row['seed'] for row in rows[:len(added_donors)]]:
        raise ValueError('student practice does not contain a prefix of the expected donors')
    if len(added_donors) == len(rows) and 'student_branch_seconds' in checkpoint_data:
        return
    started = perf_counter()
    evaluator = BranchPractice(base['source'])
    try:
        for row in rows[len(added_donors):]:
            added = evaluator.donor(row, student=True)
            checkpoint_data['contexts'].extend(added)
            checkpoint_data['donors'].append([trace_path.name, row['seed']])
            _write(checkpoint, checkpoint_data)
            print('student', row['seed'], row['longest'],
                  [(r['leg'], len(r['outcomes']), sum(a[1] for a in r['outcomes'])) for r in added],
                  round(perf_counter()-started, 1), flush=True)
    finally:
        evaluator.close()
    checkpoint_data['student_branch_seconds'] = perf_counter()-started
    _write(checkpoint, checkpoint_data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=PRACTICE)
    parser.add_argument('--trace', type=Path, default=STUDENT_TRACE)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output', type=Path, default=RECEIPT)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument('--test-only', action='store_true')
    phase.add_argument('--record-only', action='store_true')
    phase.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    if args.baseline_only:
        output = BASELINE if args.output == RECEIPT else args.output
        record(args.base, output, HELD_COUNT, HELD_SCHEDULE)
        return
    if not args.test_only:
        record(args.base, args.trace, TRAIN_COUNT, TRAIN_SCHEDULE)
        if args.record_only:
            return
        aggregate(args.base, args.trace, args.checkpoint)
    record(args.checkpoint, args.output, HELD_COUNT, HELD_SCHEDULE)


if __name__ == '__main__':
    main()
