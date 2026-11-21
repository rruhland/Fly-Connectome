"""Replay paired native Pong starts with the exact before and after hall score state."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from .tt3_numeric_retention import _hashes, _native
from .tt3_pong import assess_retention, paired_pong, state_sha256
from .tt4_transfer_trial import ARTIFACTS, LEGACY_MINI, _hash, _read, _write


def hall_states(receipt, donor, installation):
    """Verify every eligible score update and recover the actual carried ledger."""
    if any(receipt['geometry']['arena'][key] != value for key, value in
           [('render_mode', 'mujoco'), ('venue', 'hall')]):
        raise ValueError('expected native shaded hall qualification')
    course, = [entry for entry in receipt['courses']['first_person']
               if entry['schedule']['index'] == installation]
    run = course['runs']['rally:legacy_mini']
    points = run['points']
    if [row['seed'] for row in points] != course['schedule']['tests']:
        raise ValueError('hall course does not contain all scheduled points in order')
    initial, final = run['initial_prior'], run['final_prior']
    before, after = initial['numeric'], final['numeric']
    if before != donor['memory']['shared'] or before['coefficients'] != after['coefficients']:
        raise ValueError('hall entry or candidate coefficients differ from the carried mini state')
    roles = donor['roles']
    if roles['relevance_weights'] != initial['role_weights'] or initial['role_weights'] != final['role_weights']:
        raise ValueError('receipt does not provide comparable role state')
    if any(row['attributed_rewards'] for row in points):
        raise ValueError('hall role updates require a saved post-hall role model')
    eligible = [row for row in points if row['opportunity']]
    choices = [row['option'] for row in eligible]
    size = len(before['counts'])
    counts = np.bincount(choices, minlength=size)
    totals = np.bincount(choices, weights=[row['public_reward'] for row in eligible], minlength=size)
    if not np.array_equal(counts, np.subtract(after['counts'], before['counts'])) or not np.array_equal(
            totals, np.subtract(after['totals'], before['totals'])):
        raise ValueError('saved hall scores and counts differ from the chronological point ledger')
    if not choices or before == after:
        raise ValueError('no numeric state change to evaluate')
    states = dict(before=dict(memory=before, roles=roles), after=dict(memory=after, roles=roles),
                  count_only=dict(memory={**after, 'totals': before['totals']}, roles=roles))
    ledger = dict(schedule=course['schedule'], credited_choices=counts.tolist(), credited_scores=totals.tolist(),
                  points=[{key: row[key] for key in ('seed', 'option', 'public_reward', 'opportunity',
                                                    'player_returns', 'rival_returns')} for row in points])
    return states, ledger


def _evaluate(job):
    name, schedule, state = job
    output = _native(schedule, state)
    return name, output


def run(qualification_path, donor_path, pong_path, output_path, *, installation=0, workers=3):
    qualification = _read(qualification_path)
    if qualification['inputs_sha256']['legacy_mini'] != _hash(donor_path):
        raise ValueError('mini checkpoint does not match the hall experiment input digest')
    states, hall = hall_states(qualification, _read(donor_path), installation)
    canonical = _read(pong_path)
    schedule = canonical['schedule']
    if (len(schedule['evaluation']), schedule['evaluation_points']) != (36, 5):
        raise ValueError('expected the archived 36-match native Pong schedule')
    if state_sha256(schedule) != canonical['source']['schedule_sha256']:
        raise ValueError('Pong schedule digest differs from the archived checkpoint')
    implementation = {**_hashes(), Path(__file__).name: _hash(Path(__file__))}
    jobs = [(name, schedule, state) for name, state in states.items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        evaluated = dict(pool.map(_evaluate, jobs))
    before, after = evaluated['before']['points'], evaluated['after']['points']
    retention = assess_retention(before, after)
    retention['behavior_changes'] = {key: sum(a[key] != b[key] for a, b in zip(before, after, strict=True))
                                     for key in ('relation', 'action_sha256', 'contacts', 'score')}
    paired_control = paired_pong(evaluated['count_only']['points'], after)
    control_changes = sum(a['action_sha256'] != b['action_sha256']
                          for a, b in zip(evaluated['count_only']['points'], after, strict=True))
    receipt = dict(schema='tt5-hall-numeric-pong-retention-v1', hall=hall, schedule=schedule,
                   states=states, state_sha256={key: state_sha256(state) for key, state in states.items()},
                   runs=evaluated, retention=retention, after_minus_count_only=paired_control,
                   after_count_only_action_changes=control_changes, implementation_sha256=implementation,
                   inputs_sha256=dict(qualification=_hash(qualification_path), donor=_hash(donor_path),
                                      native_schedule=_hash(pong_path)))
    _write(output_path, receipt)
    print(json.dumps(dict(hall_updates=hall['credited_choices'], native={key: row['summary']
                     for key, row in evaluated.items()}, retained=retention['established'],
                     behavior_changes=retention['behavior_changes'], after_count_only_action_changes=control_changes)),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('qualification', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--donor', type=Path, default=LEGACY_MINI)
    parser.add_argument('--pong-schedule', type=Path, default=ARTIFACTS / 'pong-source-receipt.json.gz')
    parser.add_argument('--installation', type=int, default=0)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    torch.set_num_threads(1)
    run(args.qualification, args.donor, args.pong_schedule, args.output,
        installation=args.installation, workers=args.workers)


if __name__ == '__main__':
    main()
