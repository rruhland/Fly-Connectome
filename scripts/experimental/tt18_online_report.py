"""Summarize the fully replayed, two-game body exploration comparisons."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt18_online_trial import metrics


PRIMARY = ('carried', 'cold', 'passive', 'frozen')


def series(path):
    receipt = _read(path)
    games = receipt['games']
    rows = [point for game in games for point in game['points']]
    if len(games) != 2 or receipt.get('replayed_physics_points') != len(rows):
        raise ValueError(f'{path.name} must contain two complete replayed games')
    if len({row['seed'] for row in rows}) != len(rows):
        raise ValueError(f'{path.name} reused a point seed')
    if any(_hash(path.parent / game['checkpoint']) != game['checkpoint_sha256'] for game in games):
        raise ValueError(f'{path.name} has a mismatched player checkpoint')
    skill = metrics(rows)
    elapsed = skill['frames']
    first = skill['first_frames']
    skill.update(seconds=elapsed * receipt['control_dt'],
                 first_seconds={key: None if value is None else value * receipt['control_dt']
                                for key, value in first.items()},
                 returns_per_1000_frames=1000 * skill['total_legal_returns'] / elapsed)
    choices = [row['serve_exploration'] for row in rows if row.get('serve_exploration') is not None]
    recall = [row['basis_recall'] for row in rows if 'basis_recall' in row]
    openings = {}
    for count in (60, 120, 240):
        frame_errors = [sighting[5] for row in rows for sighting in row['optical'][:-1]][:count]
        visible = [error for error in frame_errors if error is not None]
        openings[str(count)] = dict(median_optical_model_error_m=float(np.median(visible)) if visible else None,
                                    rgb_blade_sightings=len(visible))
    return receipt, dict(receipt=path.name, sha256=_hash(path), metrics=skill, early_model_error=openings,
                         physical_replay=receipt['replayed_physics_points'],
                         rgb_replay=receipt.get('replayed_rgb_points', 0),
                         games=[{key: game[key] for key in ('seed', 'score', 'metrics', 'checkpoint',
                                                           'checkpoint_sha256')} for game in games],
                         serve_variations=choices, basis_recall=recall)


def analyze(paths, output):
    groups = {}
    provenance = {}
    targets = {}
    for path in paths:
        receipt, evidence = series(path)
        key = f"{receipt['site']}-{receipt['fixture']['plant']}"
        seeds = [item['seed'] for item in receipt['games']]
        common = dict(site=receipt['site'], fixture=receipt['fixture'], stream_seed=receipt['stream_seed'],
                      game_seeds=seeds, control_dt=receipt['control_dt'], fresh_ports=receipt['fresh_ports'],
                      donor_sha256=receipt['donor_sha256'], target_preparation_frames=receipt['target_preparation_frames'])
        if key in provenance and provenance[key] != common:
            raise ValueError(f'matched conditions for {key} use different fixtures or game streams')
        provenance[key] = common
        modes = groups.setdefault(key, {})
        if receipt['mode'] in modes:
            raise ValueError(f"duplicate {key} mode {receipt['mode']}")
        modes[receipt['mode']] = evidence
        targets.setdefault(key, set()).update(point['seed'] for item in receipt['games'] for point in item['points'])
    if len(groups) != 2 or any(not all(mode in modes for mode in PRIMARY) for modes in groups.values()):
        raise ValueError('provide both fixtures and all four primary modes')
    left, right = targets.values()
    if left & right:
        raise ValueError('the two fixtures reused a scored point schedule')
    _write(output, dict(schema='tt18-online-comparison-v1', provenance=provenance, fixtures=groups))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('receipts', nargs='+', type=Path)
    args = parser.parse_args()
    analyze(args.receipts, args.output)
