"""Play the frozen evidence metrics on paired native SO-101 points."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_branch_motor import BranchMotor, MOTORS
from .tt22_branch_motor_trial import NativePoint
from .tt22_opponent_learn import _replay
from .tt23_source_metric import CHECKPOINT, MODES, OUTPUT, PRACTICE, SourceMetricScorer

SCHEDULE = 1537465189
COUNT = 6
RECEIPT = OUTPUT / 'source-metric-live.json.gz'


def shared_prefix(rows):
    """Commands and measured optical prefixes agree before later-ranking can act."""
    first = rows[0]
    if any(len(row['shots']) > 1 for row in rows):
        if not all(len(row['shots']) > 1 for row in rows):
            raise ValueError('matched conditions reached different first later cues')
        tick = first['shots'][1]['tick']
        for row in rows:
            if row['shots'][1]['tick'] != tick or row['shots'][1]['features'] != first['shots'][1]['features']:
                raise ValueError('matched conditions disagree on the first later cue')
    else:
        tick = len(first['actions'])
        if any(len(row['actions']) != tick for row in rows):
            raise ValueError('matched conditions disagree before a later cue')
    optical = [sample for sample in first['optical'] if sample[0] <= tick]
    for row in rows:
        if (row['actions'][:tick] != first['actions'][:tick]
                or row['rival_actions'][:tick] != first['rival_actions'][:tick]
                or [sample for sample in row['optical'] if sample[0] <= tick] != optical):
            raise ValueError('matched opening inputs or commands changed before the later policy')
    return tick


def summarize(rows):
    later = [row['outcome']['legal_returns'] for row in rows]
    lengths = [row['longest'] for row in rows]
    return dict(lengths=lengths, first=sum(n >= 1 for n in later), second=sum(n >= 2 for n in later),
                third=sum(n >= 3 for n in later), near_returns=sum(later),
                alternating=sum(lengths), ten=sum(n >= 10 for n in lengths),
                twenty=sum(n >= 20 for n in lengths), wins=sum(row['outcome']['points_for'] for row in rows))


def repeat_commands(reference, learner):
    """Reuse captured RGB only when the learner independently issues every same command."""
    optical = dict(reference['optical'])
    learner.reset()
    for tick, original in enumerate(reference['actions']):
        measured = np.asarray(optical[tick]) if tick in optical else None
        if learner.command(tick, measured).tolist() != original:
            return None
    return dict(reference, shots=learner.shots, observation_reuse=True)


def run(checkpoint, output):
    documents = _read(checkpoint)
    practice = _read(PRACTICE)
    if documents['practice_sha256'] != _hash(PRACTICE) or practice['motors'] != [list(motor) for motor in MOTORS]:
        raise ValueError('physical practice archive or its motor library changed')
    seeds = np.random.default_rng(SCHEDULE).integers(2**31, size=COUNT).tolist()
    if len(set(seeds)) != COUNT or set(seeds) & {row[1] for row in practice['donors']}:
        raise ValueError('fresh physical seeds are not disjoint')
    openings = [row for row in practice['contexts'] if not row['leg']]
    later = [row for row in practice['contexts'] if row['leg']]
    actors = {name: BranchMotor(practice['source'], SourceMetricScorer(
        openings, later, documents['source'], name, documents['observations'][name])) for name in MODES}
    hashes = {'weights_sha256': _hash(checkpoint), 'practice_sha256': documents['practice_sha256']}
    result = dict(schema='tt23-source-metric-live-1', checkpoint=checkpoint.name, **hashes, schedule=SCHEDULE,
                  seeds=seeds, budget=documents['budget'], rows={name: [] for name in MODES},
                  prefix=[], verified=0)
    if output.exists():
        saved = _read(output)
        if any(saved[key] != result[key] for key in (*hashes, 'schedule', 'seeds', 'budget')):
            raise ValueError('incompatible earlier physical receipt')
        result = saved
    started = perf_counter()
    point = NativePoint(practice)
    try:
        for slot, seed in enumerate(seeds):
            modes = MODES if slot % 2 == 0 else MODES[::-1]
            for name in modes:
                if len(result['rows'][name]) > slot:
                    continue
                reference = next((rows[slot] for rows in result['rows'].values() if len(rows) > slot), None)
                row = repeat_commands(reference, actors[name]) if reference is not None else None
                if row is None:
                    row = dict(point.play(seed, actors[name]), observation_reuse=False)
                result['rows'][name].append(row)
                result['verified'] += _replay([row])
                _write(output, result)
                print(name, seed, row['longest'], row['outcome']['legal_returns'], row['observation_reuse'],
                      round(perf_counter() - started, 1), flush=True)
            if len(result['prefix']) == slot:
                point_rows = [result['rows'][name][slot] for name in MODES]
                result['prefix'].append(shared_prefix(point_rows))
            _write(output, result)
    finally:
        point.close()
    result['summary'] = {name: summarize(rows) for name, rows in result['rows'].items()}
    result['seconds'] = perf_counter() - started
    _write(output, result)
    print(result['summary'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=CHECKPOINT, type=Path)
    parser.add_argument('--output', default=RECEIPT, type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
