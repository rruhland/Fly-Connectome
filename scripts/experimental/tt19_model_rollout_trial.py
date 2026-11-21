"""Score and replay the independent factory-simulator receiver on fresh starts."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt7_flight import FlightMemory
from .tt8_service_arena import ServiceScore
from .tt19_flight import FlightLaw
from .tt19_match_player import MatchPlayer, evaluate
from .tt19_model_rollout import RolloutPlayer
from .tt19_so101 import SOURCE_REVISION
from .tt19_stroke_trial import DONOR
from .tt19_view_trial import SO101ViewArena


SEEDS = (424520913, 56883013, 1221432110)


def replay(report):
    arena = SO101ViewArena(venue='hall', strict_endline=True, width=240, height=180,
                           view_mode='wrist+fixed', max_steps=190)
    course = report.get('course', 'full')
    if course != 'full':
        from .tt19_skill_course import install_feeder
        install_feeder(arena, course)
    count = 0
    try:
        for game in report['games']:
            score = ServiceScore(first_server=-1)
            rng = np.random.default_rng(game['seed'])
            for row in game['points']:
                point = int(rng.integers(2**31))
                if row['seed'] != point or row['server'] != score.server or score.finished:
                    raise ValueError('rollout game has a mismatched seed, service or score')
                arena.reset(seed=point, server=score.server, render=False)
                for action in row['actions']:
                    arena.step(action, render=False)
                result = arena.point_result()
                if result != row['result'] or json.dumps(arena.events) != json.dumps(row['events']):
                    raise ValueError('rollout game action tape changed its physical events')
                score.record(result)
                count += 1
            if not score.finished or list(score.score) != game['score']:
                raise ValueError('rollout game replay did not reach the reported final score')
    finally:
        arena.close()
    return count


def save_checkpoint(path, folder, cache, bank_name, source, count):
    prior = _read(DONOR)
    series = prior['series'] if source else {}
    law_path = folder / 'flight-source.json.gz'
    law = FlightLaw.from_state(_read(law_path)['source_law']) if source else FlightLaw()
    response = FlightMemory('impulse', series.get('flight_history', ()), series.get('flight_impulses', ()))
    bank_path = folder / f'consolidated-{bank_name}-bank.json.gz'
    state = dict(bank=_read(bank_path)['banks'], bank_source=bank_name, practice_sha256=_hash(bank_path),
                 source=source, law=law.state(), response=response.response_matrix.tolist(),
                 catalog_sha256=_hash(cache), receiver='catalog', donor_sha256=_hash(DONOR),
                 flight_sha256=_hash(law_path) if source else None)
    prior['tt19_match'] = state
    prior['tt19_model_rollout'] = dict(schema='tt19-model-rollout-1', candidate_count=count,
                                      factory_source=SOURCE_REVISION, catalog=cache.name)
    _write(path, prior)


def load_checkpoint(path, cache):
    data = _read(path)
    saved, spec = data['tt19_match'], data['tt19_model_rollout']
    if (_hash(cache) != saved['catalog_sha256'] or spec['factory_source'] != SOURCE_REVISION
            or spec['schema'] != 'tt19-model-rollout-1'):
        raise ValueError('the loaded factory rollout model or its catalogue changed')
    with np.load(cache) as archive:
        catalog = dict(archive)
    law = FlightLaw.from_state(saved['law'])
    response = np.asarray(saved['response'])
    policy = MatchPlayer(saved['bank'], catalog, law, response, saved['receiver'])
    policy.player = RolloutPlayer(catalog, law, response, count=spec['candidate_count'])
    return policy


def replay_optical(policy, report):
    used = []
    for game in report['games']:
        candidates = [(index, row) for index, row in enumerate(game['points'])
                      if row['server'] == -1 and row['plans']]
        if not candidates:
            continue
        index, row = max(candidates, key=lambda pair: (pair[1]['result']['legal_returns'], pair[1]['result']['points_for']))
        policy.reset(-1)
        actions = [policy.command(tick, point).tolist() for tick, point in enumerate(row['optical'])]
        if actions != row['actions'] or policy.player.plans != row['plans']:
            raise ValueError('a receiving tape did not reproduce from its stored optical coordinates')
        used.append(dict(game_seed=game['seed'], point_index=index, point_seed=row['seed']))
    return used


def run(folder, cache, seeds, *, native=True, baseline=False, source=True, mode='local', count=32, tag=None):
    player = None
    name = ('model-rollout-analytic' if baseline else 'model-rollout') if tag is None else tag
    if not source and tag is None:
        name += '-nominal'
    name += '-native' if native else '-fast'

    def engine(catalog, law, response):
        nonlocal player
        player = RolloutPlayer(catalog, law, response, count=count)
        return player

    start = perf_counter()
    try:
        report = evaluate(folder, cache, name, list(seeds), native, 'catalog', source, mode,
                          engine=None if baseline else engine, endpoints=False)
        report['seconds'] = perf_counter() - start
        report['evaluation'] = 'analytic' if baseline else 'separate calibrated MuJoCo'
        report['internal_rollouts'] = 0 if baseline else player.evaluated
        report['max_nominal_candidates'] = 0 if baseline else count
    finally:
        if player is not None:
            player.close()
    report['replayed_points'] = replay(report)
    if not baseline:
        checkpoint = folder / f'match-{name}-player.json.gz'
        save_checkpoint(checkpoint, folder, cache, mode, source, count)
        report['checkpoint'] = checkpoint.name
        report['checkpoint_sha256'] = _hash(checkpoint)
        restored = load_checkpoint(checkpoint, cache)
        try:
            report['optical_command_replays'] = replay_optical(restored, report)
        finally:
            restored.player.close()
    _write(folder / f'match-{name}.json.gz', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--bank', default='local')
    parser.add_argument('--nominal', action='store_true')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--tag')
    parser.add_argument('--count', type=int, default=32)
    args = parser.parse_args()
    if not 8 <= args.count <= 64:
        parser.error('use 8 to 64 shortlisted candidates')
    run(args.folder, args.cache, args.seeds, native=not args.fast, baseline=args.baseline, source=not args.nominal,
        mode=args.bank,
        count=args.count, tag=args.tag)
