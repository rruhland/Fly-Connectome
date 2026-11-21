"""Summarize complete games and audit matched openings and changed-control intervals."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt14_gain_trial import load_map


SUITES = (
    ('gantry', 'coupling', ('hybrid', 'fixed', 'remap', 'refit')),
    ('five', 'coupling', ('hybrid', 'fixed', 'remap', 'refit')),
    ('gantry', 'gain', ('hybrid', 'fixed')),
    ('five', 'gain', ('hybrid', 'fixed')),
    ('gantry', 'stable', ('hybrid', 'fixed')),
    ('five', 'stable', ('hybrid', 'fixed')),
    ('gantry', 'coupling-check', ('hybrid', 'fixed', 'remap', 'refit')),
)


def phase_counts(rows):
    own = [row for row in rows if row['server'] == 1]
    rival = [row for row in rows if row['server'] == -1]
    return dict(starts=len(rows), points=sum(row['points_for'] for row in rows),
                returns=sum(row['player_returns'] for row in rows),
                first_returns=sum(row['player_returns'] > 0 for row in rows),
                sustained=sum(row['summary']['sustained'] for row in rows),
                own_serves=len(own), legal_own_serves=sum(row['evaluator']['legal_serve'] for row in own),
                rival_serves=len(rival), first_on_rival_serve=sum(row['player_returns'] > 0 for row in rival),
                second_on_rival_serve=sum(row['player_returns'] >= 2 for row in rival),
                switches=sum(row['model_used'] != row['choice']['active'] for row in rows))


def attachment_errors(source, checkpoint, actual):
    original, _ = load_map(source)
    active, transform = load_map(checkpoint)
    original_jac = original.jacobian(np.zeros(original.action_size))
    expected = original_jac @ actual
    learned = active.jacobian(np.zeros(active.action_size)) @ transform
    return dict(position_relative=float(np.linalg.norm((expected - learned)[:3]) / np.linalg.norm(expected[:3])),
                pitch_relative=float(np.linalg.norm((expected - learned)[3]) / np.linalg.norm(expected[3])))


def summarize(prefix, source, receipt):
    games = []
    timings = {kind: [] for kind in ('skipped', 'evaluated')}
    for index, game in enumerate(receipt['games']):
        raw = game['points']
        changed = [row for row in raw if row['changed']]
        original = [row for row in raw if not row['changed']]
        for row in raw:
            elapsed = row['update_cpu_ns'] / 1e6
            timings['evaluated' if 'candidate' in row['choice'] else 'skipped'].append(elapsed)
        if index == 0:
            stage = 'stable' if receipt['profile'] == 'stable' else 'changed'
        else:
            stage = 'restored'
        checkpoint = prefix.with_name(f'{prefix.name}-{stage}-player.json.gz')
        if _hash(checkpoint) != game['checkpoint']:
            raise ValueError(f'checkpoint does not match receipt {checkpoint}')
        transform = np.asarray(receipt['matrix'])
        actual = transform if stage == 'changed' else np.eye(len(transform))
        recovered = None
        if stage == 'restored':
            recovered = next((i for i, row in enumerate(original, 1) if row['choice']['active'] == 0), None)
        games.append(dict(score=game['score'], metrics=game['metrics'], changed=phase_counts(changed),
                          original=phase_counts(original), active=game['active_kind'],
                          original_map_selected_by_restored_start=recovered,
                          checkpoint=checkpoint.name, map_error=attachment_errors(source, checkpoint, actual)))
    return dict(receipt=prefix.name+'.json.gz', games=games, replays=receipt['replayed_points'],
                update_cpu_ms={kind: dict(count=len(times), median=float(np.median(times)))
                               for kind, times in timings.items() if times})


def run(folder, sources):
    results = {}
    for body, profile, modes in SUITES:
        receipts, summary = {}, {}
        for mode in modes:
            prefix = folder / f'gain-{body}-{profile}-{mode}'
            receipt = _read(prefix.with_suffix('.json.gz'))
            receipts[mode] = receipt
            summary[mode] = summarize(prefix, sources / f'live-{body}-remap-player.json.gz', receipt)
        baseline = receipts['fixed']
        opening = baseline['games'][0]['points']
        if profile != 'stable':
            opening = opening[:baseline['switch']]
        equal = {}
        for mode, receipt in receipts.items():
            if any(receipt[key] != baseline[key] for key in ('arena', 'seed', 'start', 'matrix', 'switch')):
                raise ValueError(f'game protocol differs in {body}/{profile}/{mode}')
            rows = receipt['games'][0]['points']
            equal[mode] = sum(left['action_sha256'] == right['action_sha256']
                              for left, right in zip(opening, rows))
        results[f'{body}-{profile}'] = dict(modes=summary, exact_opening_tapes=equal, opening_starts=len(opening),
                                               seed=baseline['seed'], perturbation_seed=baseline['perturbation_seed'])
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--sources', type=Path, default=Path('artifacts/tt13'))
    args = parser.parse_args()
    _write(args.folder / 'gain-summary.json.gz', run(args.folder, args.sources))
