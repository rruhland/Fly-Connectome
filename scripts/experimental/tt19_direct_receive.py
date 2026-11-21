"""Test zero-practice optical receiving with the published factory rollout player."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .tt4_transfer_trial import _hash, _read, _write
from .tt7_flight import FlightMemory
from .tt19_flight import FlightLaw
from .tt19_model_rollout import RolloutPlayer
from .tt19_stroke_trial import DONOR
from .tt19_view import ViewTransport
from .tt19_view_trial import SO101ViewArena


MODES = ('gantry', 'nominal')
SEED = 992640773
ARENA = dict(venue='hall', strict_endline=True, width=240, height=180,
             view_mode='wrist+fixed', max_steps=190, detail='preview')


def previous_seeds(folder):
    """Read evaluator seed receipts, leaving stored controllers and banks unopened."""
    seen = set()

    def inspect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('seed', 'point_seed') and isinstance(item, int):
                    seen.add(item)
                elif key not in ('actions', 'events', 'optical', 'optical_trace', 'banks'):
                    inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)

    for path in folder.rglob('*.json.gz'):
        if '-player.' not in path.name and '-bank.' not in path.name and not path.name.startswith('direct-receive'):
            inspect(_read(path))
    return seen


def make_players(cache, source):
    with np.load(cache) as archive:
        catalog = dict(archive)
    old = _read(DONOR)['series']
    law = FlightLaw.from_state(_read(source)['source_law'])
    carried = FlightMemory('impulse', old['flight_history'], old['flight_impulses']).response_matrix
    nominal = FlightMemory('impulse').response_matrix
    return (RolloutPlayer(catalog, law, carried, count=32),
            RolloutPlayer(catalog, FlightLaw(), nominal, count=32))


def play(arena, player, seed, frames=None, reference=None):
    arena.reset(seed=seed, server=-1, render=False)
    tracker = ViewTransport('wrist+fixed').tracker()
    player.reset()
    actions, optical = [], []
    shared = 0
    same = reference is not None
    for tick in range(arena.max_steps):
        if same and tick < len(reference):
            rgb = frames[tick]
            shared += 1
        else:
            rgb = arena.render_cameras()
            if reference is None:
                frames.append(rgb)
        tracker.update(rgb)
        ball = tracker.ball
        position = ball.position if ball is not None and ball.observed else None
        optical.append(None if position is None else np.asarray(position).tolist())
        action = player.command(tick, position, -1)
        actions.append(action.tolist())
        if same and (tick >= len(reference) or actions[-1] != reference[tick]):
            same = False
        _, _, done, timeout, _ = arena.step(action, render=False)
        if done or timeout:
            break
    result = arena.point_result()
    returns = [int(event['step']) for event in arena.events
               if event['kind'] == 'legal_return' and event.get('side') == 1]
    if len(returns) != result['legal_returns']:
        raise ValueError('legal-return events disagree with the referee')
    return dict(seed=seed, server=-1, frames=len(actions), actions=actions, optical=optical,
                rgb=tracker.digest.hexdigest(), result=result, events=arena.events, plans=player.plans,
                legal_return_frames=returns, shared_rgb_frames=shared)


def batch(cache, source, schedule):
    arena = SO101ViewArena(**ARENA)
    players = make_players(cache, source)
    pairs = []
    try:
        for index, seed in schedule:
            order = (0, 1) if index % 2 == 0 else (1, 0)
            first, second = order
            images = []
            left = play(arena, players[first], seed, images)
            right = play(arena, players[second], seed, images, left['actions'])
            if left['optical'][:right['shared_rgb_frames']] != right['optical'][:right['shared_rgb_frames']]:
                raise ValueError('identical RGB prefixes produced different optical positions')
            rows = {MODES[first]: left, MODES[second]: right}
            pairs.append(dict(index=index, seed=seed, **rows))
            print(index + 1, seed, [(mode, rows[mode]['result']['legal_returns'],
                                    rows[mode]['result']['points_for']) for mode in MODES], flush=True)
    finally:
        for player in players:
            player.close()
        arena.close()
    return pairs


def verify(report):
    arena = SO101ViewArena(**report['arena'])
    count = 0
    try:
        for pair in report['pairs']:
            for mode in MODES:
                row = pair[mode]
                arena.reset(seed=pair['seed'], server=-1, render=False)
                for action in row['actions']:
                    arena.step(action, render=False)
                if (arena.point_result() != row['result'] or
                        json.dumps(arena.events) != json.dumps(row['events'])):
                    raise ValueError(f"physical replay changed for {mode} point {pair['seed']}")
                count += 1
    finally:
        arena.close()
    return count


def precision(pairs, getter):
    values = {mode: [bool(getter(row[mode])) for row in pairs] for mode in MODES}
    counts = {mode: sum(outcomes) for mode, outcomes in values.items()}
    n = len(pairs)
    rates = {}
    for mode in MODES:
        interval = binomtest(counts[mode], n).proportion_ci(.95, method='wilson')
        rates[mode] = dict(count=counts[mode], total=n, percent=100 * counts[mode] / n,
                           wilson95=[100 * interval.low, 100 * interval.high])
    only_gantry = sum(a and not b for a, b in zip(*values.values(), strict=True))
    only_nominal = sum(b and not a for a, b in zip(*values.values(), strict=True))
    total = only_gantry + only_nominal
    paired_p = 1. if not total else binomtest(only_gantry, total).pvalue
    return dict(rates=rates, gantry_only=only_gantry, nominal_only=only_nominal, ties=n - total,
                risk_difference_pp=100 * (only_gantry - only_nominal) / n, exact_p=paired_p)


def summarize(pairs):
    result = dict(first_return=precision(pairs, lambda row: row['legal_return_frames']),
                  point_win=precision(pairs, lambda row: row['result']['points_for']),
                  second_return=precision(pairs, lambda row: len(row['legal_return_frames']) >= 2))
    for mode in MODES:
        timer = 0
        first_live = None
        times = []
        for point in pairs:
            row = point[mode]
            if row['legal_return_frames']:
                frame = row['legal_return_frames'][0]
                times.append(frame)
                if first_live is None:
                    first_live = timer + frame
            timer += row['frames']
        result[mode] = dict(played_frames=timer, first_success_elapsed_frames=first_live,
                            frames_to_first_return=times, median_success_frames=float(np.median(times)) if times else None,
                            success_iqr=list(np.quantile(times, [.25, .75])) if times else None)
    common = [point['gantry']['legal_return_frames'][0] - point['nominal']['legal_return_frames'][0]
              for point in pairs if all(point[mode]['legal_return_frames'] for mode in MODES)]
    result['matched_success_frame_differences'] = common
    return result


def run(folder, cache, count, workers):
    source = folder / 'flight-source.json.gz'
    seeds = np.random.default_rng(SEED).integers(2**31, size=count).tolist()
    if len(set(seeds)) != count or set(seeds) & previous_seeds(folder):
        raise ValueError('the frozen receiving schedule repeats a prior point')
    files = ('tt19_direct_receive.py', 'tt19_model_rollout.py', 'tt19_catalog.py', 'tt19_view.py',
             'tt19_view_trial.py', 'tt19_flight.py', 'tt19_so101.py')
    implementation = {name: _hash(Path(__file__).with_name(name)) for name in files}
    schedule = list(enumerate(seeds))
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        tasks = [pool.submit(batch, cache, source, schedule[index::workers]) for index in range(workers)]
        pairs = sorted([row for task in tasks for row in task.result()], key=lambda row: row['index'])
    if implementation != {name: _hash(Path(__file__).with_name(name)) for name in files}:
        raise ValueError('the frozen implementation changed during evaluation')
    report = dict(schema='tt19-direct-receive-v1', schedule_seed=SEED, point_seeds=seeds, pairs=pairs, arena=ARENA,
                  plant='full-spread strict compact SO-101', seconds_per_frame=1 / 30,
                  catalogue_sha256=_hash(cache), gantry_flight_sha256=_hash(source), donor_sha256=_hash(DONOR),
                  target_practice_frames=0, target_stroke_bank=None, setup_frames=0,
                  implementation=implementation, metrics=summarize(pairs))
    report['replayed_points'] = verify(report)
    _write(folder / 'direct-receive.json.gz', report)
    print(json.dumps(report['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('catalogue', type=Path)
    parser.add_argument('--starts', type=int, default=24)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        print('replayed', verify(_read(args.verify)))
    elif not 1 <= args.workers <= args.starts:
        parser.error('choose one or more workers and at least as many starts')
    else:
        run(args.folder, args.catalogue.resolve(), args.starts, args.workers)
