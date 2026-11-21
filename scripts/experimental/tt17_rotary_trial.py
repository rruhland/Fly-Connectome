"""Replay the inherited rotary controller against camera-deadline planning."""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_memory import load
from .tt6_render import cache_hall_stereo
from .tt7_flight import FlightMemory
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_player import ServiceMemory
from .tt8_service_trial import check_tapes
from .tt10_series_trial import _restore_tracker
from .tt12_body_trial import play_point, summary, verify_game
from .tt13_rotary_trial import checkpoint, path, schedule
from .tt14_rotary_player import ReturnCorrection
from .tt17_rotary_player import DeadlinePlayer, MeasuredSpeedPlayer


START = ROOT / 'artifacts/tt16/rotary-first-base-two-player.json.gz'
SEEDS = dict(held=479001421, one=636616161, two=1465611611, three=80570108)
REPEAT_SEED = 1085973883
MODES = ('base', 'deadline')
PLAY_MODES = (*MODES, 'speed')
HELD_COUNT = 40


def restore(source, arena, mode):
    state = _read(source)
    cfg = state['rotary_return']
    carry = state['rotary_between_points']
    library, prior, response, timing = load(source)
    player_type = MeasuredSpeedPlayer if mode == 'speed' else DeadlinePlayer
    player = player_type(library.models[cfg['service_model']], prior,
                            _restore_tracker(arena.cameras, cfg['optics']['commissioned']),
                            response, timing, receiving_model=library.models[cfg['maps']['recalled']],
                            service_belief=cfg['service_belief'], predictive=True,
                            flight_memory=FlightMemory('fixed', carry['flight'], carry['impulses']),
                            service_memory=ServiceMemory(state.get('service_memory')),
                            fixed=int(cfg['profile'].partition('-')[2]),
                            correction=ReturnCorrection(**state['rotary_flight_correction']['correction']), mode=mode)
    prior.rng.bit_generator.state = deepcopy(state['rng'])
    if prior.span_memory is not None:
        span = state['motor_span']
        prior.span_memory.active = span['active']
        prior.span_memory.exposures = {int(k): count for k, count in span['exposures'].items()}
    player.role_rows = [np.asarray(row) for row in carry['roles']]
    return library, player


def record(player, arena, wiring, seed, point_seed, phase, server):
    row = play_point(player, arena, wiring, seed, point_seed, phase, server)
    # Export optical evidence after play only when it contains a second approach.
    if row['rival_returns'] > int(server == 1):
        keys = ('tick', 'blade', 'actor', 'goal', 'horizon', 'desired_pitch', 'ball', 'deadline_weighted', 'speed_weighted')
        row['optical_trace'] = [{key: frame.get(key) for key in keys} for frame in player.trace]
    return row


def run(folder, fixture, mode, source, seed_override):
    state = _read(source)
    if state.get('second_intercept_mode', mode) != mode:
        raise ValueError('resume the stored second-interception controller')
    held = fixture in ('held', 'repeat')
    count = 32 if fixture == 'repeat' else HELD_COUNT if held else 56
    seed = (REPEAT_SEED if fixture == 'repeat' else SEEDS[fixture]) if seed_override is None else seed_override
    cases = {'-1': schedule(seed, 7482, count)}
    first_server = 1 if fixture == 'two' else -1
    if not held:
        cases['1'] = schedule(seed, 7483, count)
    config = state['rotary_return']['arena']
    report = dict(schema='tt17-rotary-deadline-v1', arena=config, seed=seed, fixture=fixture, mode=mode,
                  installation_seed=config['seed'], source_sha256=_hash(source), schedule=cases,
                  points=[], practice=held)
    if not held:
        report.update(race_to=11, first_server=first_server)
    arena = TT8ServiceArena(**config)
    wiring = TT6Wiring(arena, config['seed'])
    try:
        cache_hall_stereo(arena, reflections=True)
        library, player = restore(source, arena, mode)
        score = ServiceScore(first_server=first_server)
        used = {'-1': 0, '1': 0}
        for _ in range(count):
            if held:
                _, player = restore(source, arena, mode)
            server = -1 if held else score.server
            side = str(server)
            point_seed, phase = cases[side][used[side]]
            row = record(player, arena, wiring, seed, point_seed, phase, server)
            report['points'].append(row)
            used[side] += 1
            if not held:
                score.record(row['evaluator'])
                if score.finished:
                    break
        if not held:
            if not score.finished:
                raise RuntimeError(f'{fixture} did not finish')
            saved = path(folder, f'second-{mode}-{fixture}-player')
            checkpoint(saved, source, library, player, 'recalled')
            portable = _read(saved)
            portable.update(rotary_flight_correction=state['rotary_flight_correction'], second_intercept_mode=mode)
            _write(saved, portable)
            report.update(score=score.score, finished=True, checkpoint_sha256=_hash(saved))
    finally:
        arena.close()
    report['metrics'] = {side: summary([r for r in report['points'] if str(r['server']) == side])
                         for side in cases}
    output = path(folder, f'second-{mode}-{fixture}')
    _write(output, report)
    written = _read(output)
    if held:
        check_tapes(written)
    else:
        verify_game(written)
    report['replayed_points'] = len(written['points'])
    _write(output, report)
    print(mode, fixture, report.get('score'), report['metrics'], flush=True)


def audit_resume(folder, mode, fixture):
    game = path(folder, f'second-{mode}-{fixture}')
    saved = path(folder, f'second-{mode}-{fixture}-player')
    report = _read(game)
    if _hash(START) != report['source_sha256'] or _hash(saved) != report['checkpoint_sha256']:
        raise ValueError('the replay expects the original initial and final checkpoints')
    arena = TT8ServiceArena(**report['arena'])
    wiring = TT6Wiring(arena, report['installation_seed'])
    try:
        cache_hall_stereo(arena, reflections=True)
        _, live = restore(START, arena, mode)
        for expected in report['points']:
            row = record(live, arena, wiring, report['seed'], expected['seed'], expected['phase'], expected['server'])
            if row['action_sha256'] != expected['action_sha256'] or row['deadline_frames'] != expected['deadline_frames']:
                raise ValueError(f"RGB policy replay changed at {expected['seed']}")
        _, resumed = restore(saved, arena, mode)
        seed = 429194491
        continuation = []
        for (point_seed, phase), server in zip(schedule(seed, 3249, 4), (-1, 1, -1, 1), strict=True):
            direct = record(live, arena, wiring, seed, point_seed, phase, server)
            restored = record(resumed, arena, wiring, seed, point_seed, phase, server)
            if direct != restored:
                raise ValueError(f'restored point differed at {point_seed}')
            continuation.append(dict(seed=point_seed, phase=phase, server=server, action_sha256=direct['action_sha256']))
    finally:
        arena.close()
    _write(path(folder, f'second-{mode}-{fixture}-audit'), dict(game_sha256=_hash(game), checkpoint_sha256=_hash(saved),
                                                              replayed=len(report['points']), continuation=continuation))
    print(mode, fixture, 'verified RGB game and continuation', len(report['points']), len(continuation), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    cmd = parser.add_subparsers(dest='command', required=True)
    run_args = cmd.add_parser('play')
    run_args.add_argument('fixture', choices=(*SEEDS, 'repeat'))
    run_args.add_argument('mode', choices=PLAY_MODES)
    run_args.add_argument('--resume', type=Path, default=START)
    run_args.add_argument('--seed', type=int)
    audit_args = cmd.add_parser('audit')
    audit_args.add_argument('mode', choices=PLAY_MODES)
    audit_args.add_argument('fixture', choices=('one', 'two', 'three'))
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    if args.command == 'play':
        run(args.folder, args.fixture, args.mode, args.resume, args.seed)
    else:
        audit_resume(args.folder, args.mode, args.fixture)
