"""Compare factory-proposed and briefly corrected serves on the strict SO-101 table."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt7_flight import FlightMemory
from .tt8_service_arena import ServiceScore
from .tt19_contextual import optical_placement
from .tt19_factory_serve import LOOK, ServeSimulator, propose
from .tt19_flight import FlightLaw
from .tt19_match_player import MatchPlayer
from .tt19_model_rollout import RolloutPlayer
from .tt19_so101 import SOURCE_REVISION
from .tt19_stroke_trial import DONOR
from .tt19_view import ViewTransport
from .tt19_view_trial import SO101ViewArena


ARENA = dict(venue='hall', strict_endline=True, width=240, height=180, view_mode='wrist+fixed', max_steps=190)
PRACTICE = 1038028957
UNIQUE_PRACTICE = 169682827
GAME = 930477119


class Idle:
    def reset(self):
        self.plans = []

    def command(self, tick, position, server, *, estimated_velocity=None):
        return np.zeros(6)


def world(folder):
    donor = _read(DONOR)['series']
    law = FlightLaw.from_state(_read(folder/'flight-source.json.gz')['source_law'])
    impact = FlightMemory('impulse', donor.get('flight_history', ()), donor.get('flight_impulses', ()), frozen=True)
    return law, impact.response_matrix


def policy(programs, catalog, law, impulse, *, idle=False):
    banks = {'1': dict(programs=programs, trials=[]), '-1': dict(programs=[np.zeros(13)], trials=[])}
    result = MatchPlayer(banks, catalog, law, impulse, 'catalog')
    result.player = Idle() if idle else RolloutPlayer(catalog, law, impulse, count=32)
    return result


def point(arena, player, transport, seed, server):
    tracker = transport.tracker()
    arena.reset(seed=seed, server=server, render=False)
    player.reset(server)
    commands, timeline, optics, sightings = [], [], [], []
    for tick in range(arena.max_steps):
        xyz = None
        if tick % 2 == 0 or player.phase == 'receive':
            tracker.update(arena.render_cameras())
            ball = tracker.ball
            xyz = None if ball is None or not ball.observed else list(ball.position)
            timeline.append((tick, tracker.sources.get('ball', ())))
        optics.append(xyz)
        if xyz is not None:
            sightings.append((tick, xyz))
        action = player.command(tick, xyz)
        commands.append(action.tolist())
        _, reward, done, timeout, _ = arena.step(action, render=False)
        if done or timeout:
            break
    return dict(seed=int(seed), server=int(server), actions=commands,
                action_sha256=sha256(np.asarray(commands, dtype='<f8').tobytes()).hexdigest(),
                optical=optics, sources=timeline, plans=player.player.plans, rgb=tracker.digest.hexdigest(),
                reward=reward, result=arena.point_result(), events=arena.events.copy()), sightings


def bootstrap(folder):
    simulator = ServeSimulator(FlightLaw())
    started = perf_counter()
    try:
        result = propose(simulator)
    finally:
        simulator.close()
    result.update(schema='tt19-factory-serve-proposal-v1', cad_revision=SOURCE_REVISION,
                  simulated_frames=result['simulation_tapes'] * LOOK, wall_seconds=perf_counter() - started,
                  target_practice_points=0)
    _write(folder/'factory-serve-proposals.json.gz', result)
    return result


def _replay_camera_fits(arena, transport, trials):
    for row in trials:
        tracker = transport.tracker()
        captures = {tick for tick, _ in row['sources']}
        arena.reset(seed=row['seed'], server=1, render=False)
        for tick, action in enumerate(row['actions']):
            if tick in captures:
                tracker.update(arena.render_cameras())
            arena.step(action, render=False)
        if tracker.digest.hexdigest() != row['rgb']:
            raise ValueError('the interrupted practice camera receipt diverged on replay')


def train(folder, candidates, catalog, law, impulse, *, matched=True, previous=None):
    programs = candidates['programs']
    if len(programs) != 4:
        raise ValueError('use four offline proposals with three target repeats each')
    simulator = np.asarray(candidates['outcomes'])[:, :, 1].mean(axis=0)
    previous = {} if previous is None else previous
    arena = SO101ViewArena(**ARENA)
    camera_fits = previous.get('camera_fits')
    player = policy(programs, catalog, law, impulse, idle=True)
    transport = ViewTransport('wrist+fixed', camera_fits)
    rng = np.random.default_rng(PRACTICE if matched else UNIQUE_PRACTICE)
    seeds = rng.integers(2**31, size=3 if matched else 12).tolist()
    if len(set(seeds)) != len(seeds):
        raise ValueError('the practice launch seeds were not distinct')
    schedule = []
    for round_index in range(3):
        for order, candidate in enumerate(rng.permutation(len(programs))):
            seed = seeds[round_index] if matched else seeds[4 * round_index + order]
            schedule.append((round_index + 1, seed, int(candidate)))
    trials = list(previous.get('trials', ()))
    output = practice_path(folder, matched)
    result = dict(schema='tt19-factory-serve-correction-v1', arena=ARENA, seeds=seeds, trials=trials,
                  proposal_sha256=_hash(folder/'factory-serve-proposals.json.gz'), factory_pick=0,
                  synthetic_legal_rate=simulator.tolist(), matched_repeats=matched,
                  unique_launch_seeds=len(seeds))
    try:
        if previous and (previous['proposal_sha256'] != result['proposal_sha256'] or previous['seeds'] != seeds
                         or len(trials) > len(schedule)):
            raise ValueError('the interrupted practice used different proposals or launches')
        for row, (_, seed, candidate) in zip(trials, schedule[:len(trials)], strict=True):
            if (row['seed'], row['candidate']) != (seed, candidate):
                raise ValueError('the interrupted practice changed the scheduled candidate order')
        if camera_fits is None and trials:
            _replay_camera_fits(arena, transport, trials)
        for round_index, seed, candidate in schedule[len(trials):]:
            player.serves.force = candidate
            row, sightings = point(arena, player, transport, seed, 1)
            # Give the selector actual RGB bounces and the public signed point score.
            placement = optical_placement(sightings, 1, row['reward'], FlightLaw(), None)
            row.update(candidate=candidate, round=round_index, optical_placement=placement)
            trials.append(row)
            print('target practice', len(trials), candidate, round(placement, 3),
                  row['result']['legal_serve'], row['reward'], flush=True)
            result['camera_fits'] = transport.state()
            _write(output, result)
    finally:
        arena.close()
    values = np.asarray([[row['optical_placement'] for row in trials if row['candidate'] == index]
                         for index in range(len(programs))])
    # Give a weak preference to repeatable simulator placement and require
    # direct observations to move the selected program from its prior rank.
    rank = .28 * simulator + values.mean(axis=1)
    result.update(target_practice_points=len(trials), target_practice_frames=sum(len(row['actions']) for row in trials),
                  selection=rank.tolist(), corrected_pick=int(np.argmax(rank)))
    result['replayed_points'] = replay_points(ARENA, trials)
    _write(output, result)
    return result


def replay_points(settings, rows, score=None, game_seed=None):
    arena = SO101ViewArena(**settings)
    schedule = np.random.default_rng(game_seed) if game_seed is not None else None
    try:
        for row in rows:
            if score is not None and (row['server'] != score.server or score.finished or
                                      int(schedule.integers(2**31)) != row['seed']):
                raise ValueError('the saved service or opponent schedule diverged')
            arena.reset(seed=row['seed'], server=row['server'], render=False)
            for action in row['actions']:
                arena.step(action, render=False)
            if (arena.point_result() != row['result'] or json.dumps(arena.events) != json.dumps(row['events'])
                    or sha256(np.asarray(row['actions'], dtype='<f8').tobytes()).hexdigest() != row['action_sha256']):
                raise ValueError(f"physical tape diverged at {row['seed']}")
            if score is not None:
                score.record(arena.point_result())
        return len(rows)
    finally:
        arena.close()


def held(folder, proposals, practice, catalog, law, impulse, seed, mode):
    return play_game(policy(proposals['programs'], catalog, law, impulse), practice, seed, mode)


def play_game(player, practice, seed, mode):
    arena = SO101ViewArena(**ARENA)
    transport = ViewTransport('wrist+fixed')
    candidate = practice['factory_pick' if mode == 'factory' else 'corrected_pick']
    player.serves.force = candidate
    score = ServiceScore(first_server=-1)
    rng = np.random.default_rng(seed)
    rows = []
    try:
        for _ in range(65):
            start = int(rng.integers(2**31))
            if start in practice['seeds']:
                raise ValueError('held schedule overlaps a target practice point')
            row, _ = point(arena, player, transport, start, score.server)
            score.record(row['result'])
            rows.append(row)
            print(mode, len(rows), score.score, row['server'], row['result']['legal_returns'],
                  row['result']['legal_serve'], row['result']['reason'], flush=True)
            if score.finished:
                break
    finally:
        arena.close()
        player.player.close()
    if not score.finished:
        raise ValueError('the held game did not finish')
    return dict(seed=seed, score=list(score.score), finished=score.finished, mode=mode, candidate=candidate,
                points=rows, target_practice_points=0 if mode == 'factory' else practice['target_practice_points'],
                target_practice_frames=0 if mode == 'factory' else practice['target_practice_frames'],
                returns=sum(row['result']['legal_returns'] for row in rows),
                multiple=sum(row['result']['legal_returns'] >= 2 for row in rows),
                serve_attempts=sum(row['server'] == 1 for row in rows),
                serves=sum(row['server'] == 1 and row['result']['legal_serve'] for row in rows),
                replayed_points=replay_points(ARENA, _json(rows), ServiceScore(first_server=-1), seed))


def _json(value):
    return json.loads(json.dumps(value))


def load_checkpoint(path, cache, mode):
    if mode not in ('factory', 'corrected'):
        raise ValueError(f'unknown factory server mode {mode!r}')
    checkpoint = _read(path)
    saved = checkpoint.get('tt19_factory_serve', checkpoint)
    if (saved['schema'] != 'tt19-factory-serve-player-v1' or saved['cad_revision'] != SOURCE_REVISION
            or saved['catalog_sha256'] != _hash(cache)):
        raise ValueError('the saved factory model or motion catalogue changed')
    receiver = saved['receiver']
    if receiver['candidates'] != 32:
        raise ValueError('the checkpoint does not match the qualified receiving planner')
    with np.load(cache) as archive:
        catalog = dict(archive)
    rebuilt = policy(saved['programs'], catalog, FlightLaw.from_state(receiver['law']),
                     np.asarray(receiver['response']))
    rebuilt.serves.force = saved['factory_pick' if mode == 'factory' else 'corrected_pick']
    return rebuilt


def verify_checkpoint(report_path, cache, rebind=None):
    report = _read(report_path)
    checkpoint = report_path.parent / report['checkpoint']
    if _hash(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('the checkpoint digest changed')
    if rebind is not None:
        checkpoint = rebind
    verified = []
    for game in report['games']:
        # Prefer a legal service followed by a planned reply in the same
        # attempt, which exercises both halves of the portable controller.
        index, row = max(enumerate(game['points']), key=lambda pair: (
            pair[1]['server'] == 1 and pair[1]['result']['legal_serve'] and bool(pair[1]['plans']),
            pair[1]['server'] == 1 and pair[1]['result']['legal_serve'],
            bool(pair[1]['plans'])))
        player = load_checkpoint(checkpoint, cache, game['mode'])
        try:
            player.reset(row['server'])
            commands = [player.command(tick, observed).tolist()
                        for tick, observed in enumerate(row['optical'])]
            if commands != row['actions'] or player.player.plans != row['plans']:
                raise ValueError('saved checkpoint diverged from the held optical tape')
            verified.append(dict(mode=game['mode'], attempt=index + 1, point_seed=row['seed'],
                                 server=row['server'], commands=len(commands), plans=len(row['plans'])))
        finally:
            player.player.close()
    if rebind is None:
        report['checkpoint_optical_replay'] = verified
        _write(report_path, report)
    else:
        name = rebind.name.removesuffix('.json.gz').removesuffix('-player')
        output = rebind.with_name(f'{name}-rebind.json.gz')
        _write(output, dict(schema='tt19-factory-serve-rebind-v1', checkpoint=checkpoint.name,
                            checkpoint_sha256=_hash(checkpoint), source=report_path.name, source_sha256=_hash(report_path),
                            catalog_sha256=_hash(cache), matches=verified))
    return verified


def practice_path(folder, matched):
    name = 'factory-serve-practice' if matched else 'factory-serve-distinct-practice'
    return folder / f'{name}.json.gz'


def save_checkpoint(path, candidates, practice, law, impulse, cache):
    payload = dict(schema='tt19-factory-serve-player-v1', programs=candidates['programs'],
                   factory_pick=practice['factory_pick'], corrected_pick=practice['corrected_pick'],
                   receiver=dict(law=law.state(), response=impulse.tolist(), candidates=32),
                   catalog_sha256=_hash(cache), cad_revision=SOURCE_REVISION)
    if path.exists() and _read(path) == payload:
        return
    _write(path, payload)


def run(folder, cache, seed, assessment, *, matched=True, practice_only=False):
    proposal_path = folder/'factory-serve-proposals.json.gz'
    candidates = _read(proposal_path) if proposal_path.exists() else bootstrap(folder)
    with np.load(cache) as archive:
        catalog = dict(archive)
    law, impulse = world(folder)
    practice_file = practice_path(folder, matched)
    practice = _read(practice_file) if practice_file.exists() else {}
    if 'target_practice_points' not in practice or 'selection' not in practice:
        practice = train(folder, candidates, catalog, law, impulse, matched=matched, previous=practice)
    if practice['proposal_sha256'] != _hash(proposal_path) or practice['target_practice_points'] != 12:
        raise ValueError('the target practice budget or source differs')
    stem = 'factory-serve' if seed == GAME else f'factory-serve-{seed}'
    if not matched:
        stem += '-distinct'
    checkpoint = folder/f'{stem}-player.json.gz'
    if practice_only:
        save_checkpoint(checkpoint, candidates, practice, law, impulse, cache)
        print('practice selection', practice['corrected_pick'], practice['selection'], flush=True)
        return
    start = perf_counter()
    report = dict(schema='tt19-factory-serve-held-v1', arena=ARENA, strict_endline=True, course='full', native=True,
                  seed=seed, cad_revision=SOURCE_REVISION, proposal_sha256=_hash(proposal_path),
                  practice_sha256=_hash(practice_file), practice_source=practice_file.name,
                  catalog_sha256=_hash(cache), donor_sha256=_hash(DONOR),
                  flight_sha256=_hash(folder/'flight-source.json.gz'), receiver_parameters=dict(
                      law=law.state(), response=impulse.tolist(), candidates=32), games=[])
    output = folder/f'{stem}-held.json.gz'
    for mode in ('factory', 'corrected') if assessment == 'both' else (assessment,):
        game = held(folder, candidates, practice, catalog, law, impulse, seed, mode)
        report['games'].append(game)
        report['wall_seconds'] = perf_counter() - start
        _write(output, report)
    save_checkpoint(checkpoint, candidates, practice, law, impulse, cache)
    report['checkpoint'] = checkpoint.name
    report['checkpoint_sha256'] = _hash(checkpoint)
    _write(output, report)
    print('checkpoint optical replay', verify_checkpoint(output, cache), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, nargs='?', default=Path('artifacts/tt19'))
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--seed', type=int, default=GAME)
    parser.add_argument('--assessment', choices=('both', 'factory', 'corrected'), default='both')
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--rebind', type=Path)
    parser.add_argument('--distinct-practice', action='store_true')
    parser.add_argument('--practice-only', action='store_true')
    args = parser.parse_args()
    catalog = args.catalog or args.folder/'factory-dry-catalog.npz'
    if args.verify is not None:
        print(verify_checkpoint(args.verify, catalog, args.rebind))
    else:
        run(args.folder, catalog, args.seed, args.assessment, matched=not args.distinct_practice,
            practice_only=args.practice_only)


if __name__ == '__main__':
    main()
