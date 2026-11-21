"""Compare real rival return opportunities and strict games under matched play."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt8_service_arena import ServiceScore
from .tt19_model_rollout_trial import load_checkpoint
from .tt19_view import ViewTransport
from .tt20_fixture import TT20Arena
from .tt20_opponent import install_opponent


SOURCE = ROOT / 'artifacts/tt19'
LEARNER = SOURCE / 'match-model-rollout-native-player.json.gz'
ARCHIVE = SOURCE / 'match-model-rollout-extra-native.json.gz'
CATALOGUE = SOURCE / 'factory-dry-catalog.npz'
MODES = ('original', 'search')
MATCHED_SETTINGS = dict(mount='pedestal', wrist_optics='legacy', marked=True, strict_endline=True,
                        width=240, height=180, max_steps=190)
GAME_SETTINGS = dict(MATCHED_SETTINGS, mount='clamp', wrist_optics='outboard')
GAME_SEED = 724502817


def first_strike(events):
    return next((event for event in events if event['kind'] == 'contact' and event['side'] == 1), None)


def launches(source, each):
    rows = [row for game in _read(source)['games'] for row in game['points']]
    wanted = {side: 0 for side in (-1, 1)}
    selected = []
    for row in rows:
        side = row['server']
        outgoing = row['result']['legal_serve'] if side == 1 else row['result']['legal_returns'] > 0
        if outgoing and wanted[side] < each:
            strike = first_strike(row['events'])
            if strike is None:
                raise ValueError('a recorded legal launch has no contact')
            wanted[side] += 1
            selected.append(dict(seed=row['seed'], server=side, prefix=strike['step'],
                                 actions=row['actions'][:strike['step']], optical=row['optical'][:strike['step']]))
    if min(wanted.values()) != each:
        raise ValueError('the recorded source lacks the requested paired launch roles')
    return selected


def play(arena, learner, opponent, seed, role, archive=None):
    arena.reset(seed=seed, server=role, render=False)
    learner.reset(role)
    tracker = ViewTransport('wrist+fixed').tracker()
    actions, ball_history = [], []
    for tick in range(arena.max_steps):
        point = None
        if archive is not None and tick < archive['prefix']:
            point = archive['optical'][tick]
        elif tick % 2 == 0 or learner.phase == 'receive':
            tracker.update(arena.render_cameras())
            if tracker.ball is not None and tracker.ball.observed:
                point = tracker.ball.position
        ball_history.append(None if point is None else np.asarray(point).tolist())
        action = learner.command(tick, point)
        if archive is not None and tick < archive['prefix'] and action.tolist() != archive['actions'][tick]:
            raise ValueError('the loaded learner changed the archived opening motion')
        actions.append(action.tolist())
        _, _, terminated, truncated, _ = arena.step(action, render=False)
        if terminated or truncated:
            break
    return dict(seed=seed, server=role, actions=actions, optical=ball_history,
                result=arena.point_result(), events=arena.events, rgb=tracker.digest.hexdigest(),
                rival_search=[] if opponent is None else opponent.decisions)


def run_mode(mode, selection, *, matched=True, game=True):
    sections = {}
    learner = load_checkpoint(LEARNER, CATALOGUE)
    start = perf_counter()
    try:
        if matched:
            arena = TT20Arena(**MATCHED_SETTINGS)
            rival = install_opponent(arena) if mode == 'search' else None
            try:
                rows = []
                for launch in selection:
                    row = play(arena, learner, rival, launch['seed'], launch['server'], launch)
                    rows.append(row)
                    print('matched', mode, launch['seed'], row['result']['rival_returns'],
                          row['result']['legal_returns'], flush=True)
                sections['matched'] = rows
            finally:
                arena.close()
        if game:
            arena = TT20Arena(**GAME_SETTINGS)
            rival = install_opponent(arena) if mode == 'search' else None
            try:
                score = ServiceScore(first_server=-1)
                rng = np.random.default_rng(GAME_SEED)
                rows = []
                for _ in range(70):
                    point = int(rng.integers(2**31))
                    row = play(arena, learner, rival, point, score.server)
                    rows.append(row)
                    score.record(row['result'])
                    if score.finished:
                        break
                if not score.finished:
                    raise ValueError('the regulation score did not finish in 70 points')
                sections['game'] = dict(seed=GAME_SEED, score=list(score.score), points=rows)
                print('full game', mode, score.score, flush=True)
            finally:
                arena.close()
    finally:
        learner.player.close()
    return dict(sections, wall_seconds=perf_counter() - start)


def metrics(rows):
    for_learner = [row['result']['legal_returns'] for row in rows]
    for_rival = [row['result']['rival_returns'] for row in rows]
    exchange = [own + rival for own, rival in zip(for_learner, for_rival, strict=True)]
    qualifying = [row for row in rows if any(event['kind'] in ('legal_serve', 'legal_return')
                                             and event.get('side') == 1 for event in row['events'])]
    return dict(points=len(rows), legal_learner_serves=sum(row['server'] == 1 and row['result']['legal_serve']
                                                         for row in rows),
                legal_learner_returns=sum(for_learner), legal_rival_returns=sum(for_rival),
                incoming_reply_opportunities=len(qualifying),
                replied_opportunities=sum(any(event['kind'] == 'legal_return' and event.get('side') == -1
                                              for event in row['events'][
                                                  next(i for i, event in enumerate(row['events'])
                                                       if event['kind'] in ('legal_serve', 'legal_return')
                                                       and event.get('side') == 1) + 1:])
                                          for row in qualifying),
                points_with_two_learner_returns=sum(count >= 2 for count in for_learner),
                points_with_two_legal_rally_shots=sum(count >= 2 for count in exchange),
                points_with_three_legal_rally_shots=sum(count >= 3 for count in exchange),
                mean_legal_rally_shots=sum(exchange) / len(rows), maximum_legal_rally_shots=max(exchange),
                longest_point_frames=max(len(row['actions']) for row in rows))


def verify_section(rows, mode, settings, game=None):
    arena = TT20Arena(**settings)
    if mode == 'search':
        install_opponent(arena)
    score = ServiceScore(first_server=-1) if game is not None else None
    rng = np.random.default_rng(GAME_SEED) if game is not None else None
    try:
        for row in rows:
            if score is not None and (score.finished or score.server != row['server']
                                      or int(rng.integers(2**31)) != row['seed']):
                raise ValueError('the score or scheduled ball changed on replay')
            arena.reset(seed=row['seed'], server=row['server'], render=False)
            for action in row['actions']:
                arena.step(action, render=False)
            if arena.point_result() != row['result'] or json.dumps(arena.events) != json.dumps(row['events']):
                raise ValueError(f"physical {mode} replay changed on point {row['seed']}")
            if score is not None:
                score.record(arena.point_result())
        if score is not None and (not score.finished or list(score.score) != game):
            raise ValueError('the same scorer did not reproduce the full game')
    finally:
        arena.close()
    return len(rows)


def verify(report):
    verified = 0
    for mode in MODES:
        entry = report['variants'][mode]
        if 'matched' in entry:
            verified += verify_section(entry['matched'], mode, report['settings']['matched'])
        if 'game' in entry:
            verified += verify_section(entry['game']['points'], mode, report['settings']['game'],
                                       entry['game']['score'])
    return verified


def run(output, count, workers, section):
    use_matched = section != 'game'
    use_game = section != 'matched'
    selection = launches(ARCHIVE, count) if use_matched else []
    files = ('tt20_opponent.py', 'tt20_fixture.py', 'tt20_opponent_trial.py')
    frozen = {name: _hash(Path(__file__).with_name(name)) for name in files}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        jobs = {mode: pool.submit(run_mode, mode, selection, matched=use_matched, game=use_game) for mode in MODES}
        variants = {mode: job.result() for mode, job in jobs.items()}
    if frozen != {name: _hash(Path(__file__).with_name(name)) for name in files}:
        raise ValueError('the fixture or opponent changed during qualification')
    if use_matched:
        for a, b in zip(*(variant['matched'] for variant in variants.values()), strict=True):
            if first_strike(a['events']) is None or first_strike(a['events']) != first_strike(b['events']):
                raise ValueError('the opponent changed the initial learner strike in a matched pair')
    report = dict(schema='tt20-physical-opponent-v1', variants=variants,
                  settings=dict(matched=MATCHED_SETTINGS, game=GAME_SETTINGS),
                  learner_sha256=_hash(LEARNER), catalogue_sha256=_hash(CATALOGUE),
                  archived_launches_sha256=_hash(ARCHIVE) if use_matched else None,
                  implementation=frozen)
    for variant in variants.values():
        variant['metrics'] = {}
        if use_matched:
            variant['metrics']['matched'] = metrics(variant['matched'])
        if use_game:
            variant['metrics']['game'] = metrics(variant['game']['points'])
    report['physical_replays'] = verify(report)
    _write(output, report)
    print({mode: row['metrics'] for mode, row in variants.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--per-role', type=int, default=6)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--section', choices=('both', 'matched', 'game'), default='both')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print('replayed', verify(_read(args.output)))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        run(args.output, args.per_role, args.workers, args.section)
