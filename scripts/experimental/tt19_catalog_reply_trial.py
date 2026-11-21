"""Compare an early, wider second-shot catalogue against the same native match opener."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import ROOT, _read, _write
from .tt19_catalog import Player, dry_catalog
from .tt19_match_player import evaluate
from .tt19_model_rollout_trial import replay


SEEDS = (4531227, 29175201, 17634507)


def reply_metrics(game):
    points = game['points']
    return dict(
        score=game['score'],
        points=len(points),
        plans=sum(len(row['plans']) for row in points),
        points_planned=sum(bool(row['plans']) for row in points),
        own_legal_serves=game['serves'],
        legal_returns=game['returns'],
        legal_rival_replies=sum(row['result']['rival_returns'] for row in points),
        multiple_return_points=game['multiple'],
        own_serve_and_return=sum(row['server'] == 1 and row['result']['legal_serve']
                                 and row['result']['legal_returns'] > 0 for row in points),
    )


def compare(base, changed):
    base_rows = {(row['seed'], row['server']): row for row in base['points']}
    metrics = dict(shared_points=0, matched_before_plan=0, numerically_matched_before_plan=0, earlier_plans=0,
                   matched_plan_ticks=[], baseline=reply_metrics(base), wider_early=reply_metrics(changed))
    for row in changed['points']:
        previous = base_rows.get((row['seed'], row['server']))
        if previous is None:
            continue
        metrics['shared_points'] += 1
        before = previous['plans'][0]['tick'] if previous['plans'] else len(previous['actions'])
        after = row['plans'][0]['tick'] if row['plans'] else len(row['actions'])
        boundary = min(before, after, len(previous['actions']), len(row['actions']))
        if previous['actions'][:boundary] == row['actions'][:boundary]:
            metrics['matched_before_plan'] += 1
        if np.allclose(previous['actions'][:boundary], row['actions'][:boundary], atol=1e-7, rtol=0.):
            metrics['numerically_matched_before_plan'] += 1
        if previous['plans'] and row['plans']:
            metrics['earlier_plans'] += after < before
            metrics['matched_plan_ticks'].append([before, after])
    return metrics


def add_paired_metrics(trial, baseline_path):
    baseline = _read(baseline_path)
    seeds = [game['seed'] for game in trial['games']]
    reference = {game['seed']: game for game in baseline['games'] if game['finished']}
    baseline_games = [reference[seed] for seed in seeds]
    source = json.dumps(baseline_games, separators=(',', ':'), allow_nan=False).encode()
    trial['paired_baseline_games_sha256'] = sha256(source).hexdigest()
    trial['baseline_catalog_sha256'] = baseline['cache_sha256']
    trial['replayed_points'] = replay(trial)
    trial['baseline_replayed_points'] = replay({'games': baseline_games})
    trial['paired'] = [dict(seed=game['seed'], **compare(reference[game['seed']], game)) for game in trial['games']]


def run(folder, cache, output, seeds):
    folder = folder.resolve()
    baseline_path = folder / 'match-full-from-wide.json.gz'
    baseline = _read(baseline_path)
    reference = {game['seed']: game for game in baseline['games'] if game['finished']}
    if not all(seed in reference for seed in seeds):
        raise ValueError('the native baseline has not completed all paired seeds')
    if not (baseline['native'] and baseline['strict_endline'] and baseline['bank_source'] == 'full-from-wide'
            and baseline['receiver'] == 'memory' and baseline['course'] == 'full' and baseline['source']):
        raise ValueError('expected the full native strict memory-opening match source')
    with np.load(cache) as atlas:
        size = len(atlas['poses'])
    start = perf_counter()
    with TemporaryDirectory(prefix='tt19-catalog-reply-') as run_dir:
        working = Path(run_dir)
        for name in ('consolidated-full-from-wide-bank.json.gz', 'flight-source.json.gz'):
            (working / name).symlink_to(folder / name)
        trial = evaluate(working, cache, 'reply-wide-native', list(seeds), True, 'memory', True,
                         'full-from-wide', engine=lambda atlas, law, response: Player(atlas, law, response, 'reply'),
                         endpoints=False)
    trial['method'] = 'wide reply after memory opening'
    trial['catalog_count'] = size
    trial['wall_seconds'] = perf_counter() - start
    add_paired_metrics(trial, baseline_path)
    _write(output, trial)
    return trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, nargs='?', default=ROOT / 'artifacts' / 'tt19')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--build-count', type=int, default=None)
    args = parser.parse_args()
    cache = args.cache or (args.folder / 'catalog-reply-wide.npz')
    if args.build_count is not None:
        dry_catalog(cache, args.build_count, wide=True)
    run(args.folder, cache, args.output or (args.folder / 'catalog-reply-trial.json.gz'), args.seeds)


if __name__ == '__main__':
    main()
