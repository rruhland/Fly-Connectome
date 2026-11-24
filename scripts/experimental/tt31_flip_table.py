"""Transfer only the solo wrist practice's measured command delay to table play."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_action_memory import run
from .tt29_vision_replay import replay


OPENINGS = (7331, 3152, 5736, 6739)
CHECKPOINT = Path('artifacts/tt29/envelope-carry.json.gz')
OUTPUT = Path('artifacts/tt31')


def evaluate(feeds, output, practice, checkpoint):
    """Play both prospective conditions with the same frozen movement archive."""
    for group in ('cold', 'carried'):
        run(feeds, output / f'flip-table-{group}.json.gz', OPENINGS, 'left+right', 'lateral',
            checkpoint=checkpoint, no_memory=True, condition='latency', refine=True,
            size=(256, 192), camera_period=2, rated=True, motor_envelope=True,
            fast_partner=True, cached_stereo=True, seed_offset=317911, return_target=12,
            timing_checkpoint=practice if group == 'carried' else None)


def verify(feeds, output, practice, checkpoint):
    paths = {group: output / f'flip-table-{group}.json.gz' for group in ('cold', 'carried')}
    trials = {group: _read(path) for group, path in paths.items()}
    solo = _read(practice)
    lag = solo['carried_lag']
    checkpoint_digest = _hash(checkpoint)
    feeds_digest = _hash(feeds)
    for group, trial in trials.items():
        if (tuple(row['shot']['id'] for row in trial['rows']) != OPENINGS or
                trial.get('initial_checkpoint_sha256') != checkpoint_digest or
                trial.get('feed_sha256') != feeds_digest or not trial['no_memory']):
            raise ValueError(f'Unexpected playing schedule or starting state for {group}')
    cold, carried = trials['cold'], trials['carried']
    if cold['calibrated_lag'] is not None or carried['calibrated_lag'] != lag:
        raise ValueError('The table learner did not use the intended delay estimates')
    exclusions = {'calibrated_lag', 'timing_checkpoint', 'rows', 'checkpoint'}
    if any(cold[key] != carried[key] for key in cold if key not in exclusions):
        raise ValueError('Table trial compared unmatched settings')
    rows = []
    with TemporaryDirectory(prefix='tt31-flip-table-') as scratch:
        for old, new in zip(cold['rows'], carried['rows'], strict=True):
            if old['opponent_seed'] != new['opponent_seed'] or old['shot'] != new['shot']:
                raise ValueError('Table trial compared unmatched openings or seeds')
            identifier = old['shot']['id']
            measurements = {}
            for group, row in (('cold', old), ('carried', new)):
                receipt = replay(feeds, paths[group], identifier, Path(scratch) / 'audit.json')
                measurements[group] = dict(
                    returns=receipt['score']['returns'], learner=receipt['score']['learner'],
                    score=receipt['score'], exact_replay=receipt['exact_events'] and receipt['exact_score'],
                    events_sha256=receipt['events_sha256'], state_sha256=receipt['state_sha256'],
                    CAD=receipt['clearance'], process_cpu_s=row['process_cpu_s'],
                )
            rows.append(dict(id=identifier, opponent_seed=old['opponent_seed'], **measurements))
    totals = {g: {k: sum(row[g][k] for row in rows) for k in ('returns', 'learner')} for g in paths}
    report = dict(schema='tt31-flip-timing-table-1', solo_sha256=_hash(practice), carried_lag=lag,
                  measured_feed_sha256=feeds_digest, starting_checkpoint_sha256=checkpoint_digest,
                  archive_sha256={group: _hash(path) for group, path in paths.items()}, rows=rows, totals=totals)
    _write(output / 'flip-table-audit.json.gz', report)
    print('paired table totals', totals, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('feeds', type=Path)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--practice', type=Path, default=OUTPUT / 'flip-practice.json.gz')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.verify_only:
        evaluate(args.feeds, args.output, args.practice, args.checkpoint)
    verify(args.feeds, args.output, args.practice, args.checkpoint)
