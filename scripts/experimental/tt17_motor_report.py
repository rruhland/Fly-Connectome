"""Verify portable motor endpoints and summarize qualification games and motion."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt14_gain_trial import load_map
from .tt17_motor_protocol import CASES, setup_path
from .tt17_motor_trial import MODES, prefix, tally


def aggregate(rows):
    total = sum(row['clock']['total'] for row in rows)
    points = sum(row['score'][0] for row in rows)
    first = [row['clock']['first_legal_return'] for row in rows if row['clock']['first_legal_return'] is not None]
    return dict(games=len(rows), wins=sum(p['score'][0] > p['score'][1] for p in rows),
                points=points, elapsed_frames=total, points_per_1000_frames=points*1000./total,
                successful_first_games=len(first), median_first_frames_for_success=float(np.median(first)) if first else None,
                legal_returns=sum(row['metrics']['total_legal_returns'] for row in rows),
                points_with_two_returns=sum(row['metrics']['two_return_points'] for row in rows),
                rival_serve_points_with_two_returns=sum(row['rival_two_returns'] for row in rows),
                additional_dry_frames=sum(row['clock']['additional_dry'] for row in rows),
                scored_tapes=sum(row['replayed_points'] for row in rows),
                in_game_updates=sum(row['updates'] for row in rows))


def report(folder):
    fixtures = {}
    groups = {'fresh_changed': {mode: [] for mode in MODES}, 'stable': {mode: [] for mode in MODES},
              'tt14_mixers': {mode: [] for mode in MODES}}
    for case in CASES:
        path = setup_path(folder, case)
        spec = _read(path)
        group = 'stable' if spec['profile'] == 'stable' else 'tt14_mixers' if case.endswith('tt14') else 'fresh_changed'
        record = dict(profile=spec['profile'], origin=spec['origin'], setup=path.name, games={})
        audit = folder / f'motor-{case}-motion-audit.json.gz'
        if audit.exists():
            tested = _read(audit)
            if tested['setup_sha256'] != _hash(path):
                raise ValueError('independent pulse comparison used different hardware')
            record.update(motion_audit=audit.name, motion_error_m=tested['mean_error_m'],
                          motion_audit_samples=len(tested['rows']), initial_optical_fit=tested['calibration'])
        tapes = {}
        for mode in MODES:
            stem = prefix(folder, case, mode)
            raw = _read(stem.with_suffix('.json.gz'))
            endpoint = stem.with_name(stem.name + '-player.json.gz')
            game = raw['games'][0]
            rows = game['points']
            optics = sum(piece['frames'] for piece in raw['preparation'].values())
            if (tally(rows, optics, spec['control_dt']) != raw['clock'] or _hash(endpoint) != raw['checkpoint']
                    or raw['camera'] != spec['camera'] or raw['matrix'] != spec['matrix']
                    or raw['setup_sha256'] != _hash(path) or raw['replayed_points'] != len(rows)):
                raise ValueError('portable checkpoint or game receipt does not reproduce the protocol')
            model, matrix = load_map(endpoint)
            if (matrix.tolist() != raw['final_map']['transform']
                    or model.jacobian(np.zeros(model.action_size)).tolist() != raw['final_map']['camera_jacobian']):
                raise ValueError('saved model and end-of-game motor report differ')
            updates = sum(bool(row['choice'].get('adopted')) or row['choice'].get('active', row['model_used'])
                          != row['model_used'] for row in rows)
            summary = {key: raw[key] for key in ('clock', 'preparation', 'optical_fit', 'initial_map', 'final_map',
                                                'replayed_points', 'replayed_dry')}
            summary.update(score=game['score'], metrics=game['metrics'], updates=updates,
                           rival_two_returns=sum(row['server'] == -1 and row['player_returns'] >= 2 for row in rows),
                           receipt=stem.name+'.json.gz', checkpoint=endpoint.name)
            groups[group][mode].append(summary)
            record['games'][mode] = summary
            tapes[mode] = [row['action_sha256'] for row in rows]
        record['all_starts_exact_between_signed_and_bounded'] = tapes['signed'] == tapes['bounded']
        record['all_starts_exact_between_signed_and_tt15'] = tapes['signed'] == tapes['tt15']
        fixtures[case] = record
    overall = {group: {mode: aggregate(rows) for mode, rows in cohort.items()} for group, cohort in groups.items()}
    return dict(fixtures=fixtures, groups=overall)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    summary = report(args.folder)
    _write(args.folder / 'motor-summary.json.gz', summary)
    print(summary['groups'])
