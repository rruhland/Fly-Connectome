"""Measure both CAD search budgets in matched fresh native table games."""

import argparse
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt7_flight import FlightMemory
from .tt19_fast_play import TimedRolloutPlayer
from .tt19_match_player import evaluate
from .tt19_model_rollout_trial import replay
from .tt19_flight import FlightLaw
from .tt19_stroke_trial import DONOR


DEFAULT = 1221432110


def aggregate(timed):
    cpu = [call['cpu_ns'] / 1e6 for call in timed]
    wall = [call['wall_ns'] / 1e6 for call in timed]
    return dict(calls=len(timed), fallback=sum(call['fallback'] for call in timed),
                internal_trials=sum(call['full_trials'] for call in timed),
                cached_trials=sum(call['cached_trials'] for call in timed),
                internal_batches=sum(call['full_batches'] for call in timed),
                median_cpu_ms=float(np.median(cpu)) if cpu else None,
                p95_cpu_ms=float(np.quantile(cpu, .95)) if cpu else None,
                median_wall_ms=float(np.median(wall)) if wall else None,
                p95_wall_ms=float(np.quantile(wall, .95)) if wall else None,
                planning_cpu_s=sum(cpu) / 1000, planning_wall_s=sum(wall) / 1000)


def prior_cost(folder):
    initial = folder / 'context-local-practice.json.gz'
    carried = folder / 'consolidated-local-bank.json.gz'
    qualification = _read(carried)
    if qualification['source_sha256'] != _hash(initial):
        raise ValueError('the carried bank has a different practice parent')
    starting = _read(initial)['rows']
    repeats = [row for role in ('-1', '1') for row in qualification['banks'][role]['trials']]
    info = dict(search_points=len(starting), search_frames=sum(row[2] for row in starting),
                repeat_points=len(repeats), repeat_frames=sum(row[2] for row in repeats),
                initial_sha256=_hash(initial), bank_sha256=_hash(carried))
    return info, {row[0] for row in starting} | {row[1] for row in repeats}


def qualification(folder, cache, seed, adaptive, seen, prior):
    tag = 'adaptive' if adaptive else 'full'
    player = None

    def engine(catalog, law, response):
        nonlocal player
        player = TimedRolloutPlayer(catalog, law, response, adaptive=adaptive)
        return player

    with TemporaryDirectory(prefix='tt19-fast-play-') as scratch:
        directory = Path(scratch)
        for name in ('consolidated-local-bank.json.gz', 'flight-source.json.gz'):
            copyfile(folder / name, directory / name)
        wall, cpu = perf_counter(), process_time()
        try:
            report = evaluate(directory, cache, tag, [seed], True, 'catalog', True, 'local',
                              engine=engine, subset='full', endpoints=False)
            timings = player.calls
        finally:
            if player is not None:
                player.close()
        report.update(game_wall_s=perf_counter() - wall, game_cpu_s=process_time() - cpu,
                      policy=tag, additional_setup_frames=0, bank_practice=prior,
                      dry_catalog_sha256=_hash(cache), planning=aggregate(timings), timing_calls=timings)
    held = {row['seed'] for game in report['games'] for row in game['points']}
    if held & seen:
        raise ValueError('a held point repeated a bank practice seed')
    report['replayed_points'] = replay(report)
    path = folder / f'fast-play-{seed}-{tag}.json.gz'
    _write(path, report)
    print(tag, report['planning'], flush=True)
    return path, report


def matched_timing(folder, cache, queries):
    """Repeat planning on the first six observed baseline queries."""
    with np.load(cache) as archive:
        catalog = dict(archive)
    scores = _read(DONOR)['series']
    response = FlightMemory('impulse', scores['flight_history'], scores['flight_impulses']).response_matrix
    law = FlightLaw.from_state(_read(folder / 'flight-source.json.gz')['source_law'])
    players = [TimedRolloutPlayer(catalog, law, response, adaptive=adaptive)
               for adaptive in (False, True)]
    try:
        rows = []
        for number, query in enumerate(queries, 1):
            evidence = []
            # Alternate order to distribute any sequential host contention.
            order = (0, 1) if number % 2 else (1, 0)
            for index in order:
                player = players[index]
                player.history = [(t, None if xyz is None else np.array(xyz)) for t, xyz in query['optical']]
                result = player.choose(tuple(np.array(part) for part in query['estimate']))
                if index == 0 and result != query['decision']:
                    raise ValueError('the baseline did not reproduce its optical decision')
                evidence.append((index, player.calls[-1]))
            evidence = dict(evidence)
            rows.append(dict(baseline=evidence[0], adaptive=evidence[1]))
    finally:
        for player in players:
            player.close()
    return timing_comparison(rows)


def timing_comparison(rows):
    def stroke(decision):
        return None if decision is None else (decision['index'], decision['duration'])

    return dict(calls=rows, baseline=aggregate([row['baseline'] for row in rows]),
                adaptive=aggregate([row['adaptive'] for row in rows]),
                exact_decisions=sum(row['baseline']['decision'] == row['adaptive']['decision'] for row in rows),
                same_commands=sum(stroke(row['baseline']['decision']) == stroke(row['adaptive']['decision'])
                                  for row in rows))


def identical_points(full, adaptive):
    left, right = (report['games'][0]['points'] for report in (full, adaptive))
    return sum((a['seed'], a['server'], a['actions'], a['events'])
               == (b['seed'], b['server'], b['actions'], b['events']) for a, b in zip(left, right))


def run(folder, cache, seed):
    prior, seen = prior_cost(folder)
    full_path, full = qualification(folder, cache, seed, False, seen, prior)
    adaptive_path, adaptive = qualification(folder, cache, seed, True, seen, prior)
    prefix = [g['points'][0] for g in (full['games'][0], adaptive['games'][0])]
    if prefix[0]['seed'] != prefix[1]['seed'] or prefix[0]['optical'][0] != prefix[1]['optical'][0]:
        raise ValueError('the comparison opened on different RGB points')
    common = matched_timing(folder, cache, full['timing_calls'][:6])
    groups = []
    for path, report in ((full_path, full), (adaptive_path, adaptive)):
        game = report['games'][0]
        own = sum(row['server'] == 1 for row in game['points'])
        groups.append(dict(receipt=path.name, sha256=_hash(path), policy=report['policy'],
                           score=game['score'], legal_returns=game['returns'], legal_serves=game['serves'],
                           serve_attempts=own, two_return_points=game['multiple'],
                           game_frames=sum(len(row['actions']) for row in game['points']),
                           game_wall_s=report['game_wall_s'], game_cpu_s=report['game_cpu_s'],
                           planning=report['planning'], replayed_points=report['replayed_points']))
    summary = dict(schema='tt19-fast-play-v1', game_seed=seed, strict=True, native=True,
                   additional_setup_frames=0, bank_practice=prior, paired_planning=common, variants=groups)
    summary['identical_points'] = identical_points(full, adaptive)
    _write(folder / f'fast-play-{seed}-summary.json.gz', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--seed', default=DEFAULT, type=int)
    args = parser.parse_args()
    run(args.folder, args.cache.resolve(), args.seed)
