"""Aggregate matched game audits and replay the historical optical development data."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_memory import load
from .tt14_gain_report import phase_counts
from .tt15_motor_trial import archive_point, restore_memory


SUITES = (('gantry', 'coupling'), ('five', 'gain'), ('gantry', 'stable'), ('five', 'stable'))
FOLLOWUP = (('gantry', 'gain-fresh'), ('five', 'coupling-fresh'))
MODES = ('guard', 'hybrid', 'fixed', 'probe')


def report(folder):
    suites = {}
    for site, profile in (*SUITES, *FOLLOWUP):
        series = {}
        prefix = f'motor-{site}-{profile}'
        comparison = folder / f'{prefix}-fixed.json.gz'
        followup = (site, profile) in FOLLOWUP
        if followup and not comparison.is_file():
            continue
        fixed = _read(comparison)
        modes = ('hybrid', 'fixed', 'probe') if followup else MODES
        for mode in modes:
            name = f'{prefix}-{mode}'
            path = folder / f'{name}.json.gz'
            if mode == 'probe' and not path.is_file():
                continue
            receipt = _read(path)
            if any(receipt[k] != fixed[k] for k in ('arena', 'matrix', 'seed', 'start', 'switch')):
                raise ValueError(f'protocol differs across paired branches for {name}')
            metrics, times = [], []
            for i, game in enumerate(receipt['games']):
                stage = 'stable' if profile == 'stable' else 'changed' if i == 0 else 'restored'
                checkpoint = f'{name}-{stage}-player.json.gz'
                if _hash(folder / checkpoint) != game['checkpoint']:
                    raise ValueError(f'checkpoint digest differs from receipt for {name}')
                rows = game['points']
                times.extend(row['update_cpu_ns'] / 1e6 for row in rows)
                changed = [row for row in rows if row['changed']]
                original = [row for row in rows if not row['changed']]
                changes = [dict(start=j, changed=row['changed'], kind=row['choice'].get('candidate'),
                                old=row['model_used'], new=row['choice']['active'])
                           for j, row in enumerate(rows, 1) if row['choice']['active'] != row['model_used']]
                metrics.append(dict(score=game['score'], changed=phase_counts(changed), original=phase_counts(original),
                                    decisions=changes, active_kind=game['active_kind'], full_game=game['metrics'],
                                    checkpoint=checkpoint, probes=sum(r.get('probe') is not None for r in rows),
                                    probe_frames=sum(len(r['probe']['poses']) - 1 for r in rows if r.get('probe'))))
            opening = fixed['games'][0]['points']
            if profile != 'stable':
                opening = opening[:fixed['switch']]
            paired = zip(opening, receipt['games'][0]['points'])
            same_tapes = sum(left['action_sha256'] == right['action_sha256'] for left, right in paired)
            series[mode] = dict(games=metrics, opening_tapes=same_tapes, fixed_opening_starts=len(opening),
                                median_update_cpu_ms=float(np.median(times)), physical_replays=receipt['replayed_points'])
            if mode == 'probe':
                series[mode].update(reference_frames=len(receipt['reference']['poses']) - 1,
                                    optical_probe_replays=receipt['replayed_probes'])
        suites[f'{site}-{profile}'] = series
    return suites


def historical(folder, donors):
    results = {}
    for path in sorted(folder.glob('gain-*-fixed.json.gz')):
        site = path.name.split('-')[1]
        donor = donors / f'live-{site}-remap-player.json.gz'
        memory = restore_memory(_read(donor), load(donor)[0])
        records = []
        archive = _read(path)
        for game in archive['games']:
            changes = []
            for start, row in enumerate(game['points'], 1):
                update = archive_point(memory, row['actions'], row['motor_sightings'])
                if update['active'] != update['previous']:
                    changes.append(dict(start=start, was_changed=row['changed'], choice=update))
            records.append(dict(score=game['score'], changes=changes))
        results[path.name] = records
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--prior', type=Path, default=Path('artifacts/tt14'))
    parser.add_argument('--donors', type=Path, default=Path('artifacts/tt13'))
    args = parser.parse_args()
    _write(args.folder / 'motor-summary.json.gz',
           dict(fresh=report(args.folder), development=historical(args.prior, args.donors)))
