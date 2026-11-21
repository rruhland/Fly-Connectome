"""Compare optical recovery to the previously qualified blind neutral reset."""

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
from .tt13_rotary_trial import add_diagnostics, checkpoint, path, schedule
from .tt14_rotary_player import ReturnCorrection
from .tt14_rotary_trial import restore as restore_blind
from .tt15_rotary_player import BoundedReplyPlayer, FeedbackReplyPlayer, OpticalRecoveryPlayer


START = ROOT / 'artifacts/tt14/rotary-recenter-game-1-player.json.gz'
ROTARY_MODES = {'optical': OpticalRecoveryPlayer, 'bounded': BoundedReplyPlayer, 'feedback': FeedbackReplyPlayer}
MODES = ('blind', 'optical')
PLAY_MODES = ('blind', *ROTARY_MODES)
FIXTURES = (907805659, 1245582961, 793766842, 1637112805)
HELD_COUNT = 40
LIMIT = 56


def restore(start, arena, mode):
    state = _read(start)
    cfg = state['rotary_return']
    correction = ReturnCorrection(**state['rotary_flight_correction']['correction'])
    if mode == 'blind':
        return restore_blind(start, arena, 'recenter', correction)
    memory = state['rotary_between_points']
    library, prior, response, timing = load(start)
    factory = _restore_tracker(arena.cameras, cfg['optics']['commissioned'])
    player = ROTARY_MODES[mode](library.models[cfg['service_model']], prior, factory, response, timing,
                               receiving_model=library.models[cfg['maps']['recalled']],
                               service_belief=cfg['service_belief'], predictive=True,
                               flight_memory=FlightMemory('fixed', memory['flight'], memory['impulses']),
                               service_memory=ServiceMemory(state.get('service_memory')),
                               fixed=int(cfg['profile'].partition('-')[2]), correction=correction, mode=mode)
    prior.rng.bit_generator.state = deepcopy(state['rng'])
    if prior.span_memory is not None:
        span = state['motor_span']
        prior.span_memory.active = span['active']
        prior.span_memory.exposures = {int(key): count for key, count in span['exposures'].items()}
    player.role_rows = [np.asarray(row) for row in memory['roles']]
    return library, player


def fixture_name(fixture, label):
    name = 'held' if fixture == 0 else f'game-{fixture}'
    return f'{name}-{label}' if label else name


def play(folder, mode, fixture, *, start=START, seed=None, label=None):
    state = _read(start)
    if state.get('rotary_recovery_mode', mode) != mode:
        raise ValueError('resume the checkpoint with its saved recovery controller')
    config = state['rotary_return']['arena']
    held = fixture == 0
    seed = FIXTURES[fixture] if seed is None else seed
    cases = {'-1': schedule(seed, 2173, HELD_COUNT if held else LIMIT)}
    first_server = 1 if fixture == 2 else -1
    if not held:
        cases['1'] = schedule(seed, 2174, LIMIT)
    key = f'{mode}-{fixture_name(fixture, label)}'
    report = dict(schema='tt15-rotary-recovery-v1', arena=config, seed=seed, mode=mode, fixture=fixture,
                  installation_seed=config['seed'], source_sha256=_hash(start), schedule=cases,
                  points=[], practice=held)
    if not held:
        report.update(race_to=11, first_server=first_server)
    arena = TT8ServiceArena(**config)
    wiring = TT6Wiring(arena, config['seed'])
    try:
        cache_hall_stereo(arena, reflections=True)
        library, player = restore(start, arena, mode)
        score = ServiceScore(first_server=first_server)
        used = {'-1': 0, '1': 0}
        for _ in range(HELD_COUNT if held else LIMIT):
            if held:
                _, player = restore(start, arena, mode)
            server = -1 if held else score.server
            side = str(server)
            point_seed, phase = cases[side][used[side]]
            row = play_point(player, arena, wiring, seed, point_seed, phase, server)
            add_diagnostics(row, player)
            report['points'].append(row)
            used[side] += 1
            if not held:
                score.record(row['evaluator'])
                if score.finished:
                    break
        if not held:
            if not score.finished:
                raise RuntimeError(f'{key} did not finish')
            saved = path(folder, f'{key}-player')
            checkpoint(saved, start, library, player, 'recalled')
            portable = _read(saved)
            portable['rotary_flight_correction'] = state['rotary_flight_correction']
            portable['rotary_recovery_mode'] = mode
            _write(saved, portable)
            report.update(score=score.score, finished=True, checkpoint_sha256=_hash(saved))
    finally:
        arena.close()
    report['metrics'] = {side: summary([r for r in report['points'] if str(r['server']) == side])
                         for side in cases}
    output = path(folder, key)
    _write(output, report)
    written = _read(output)
    if held:
        check_tapes(written)
    else:
        verify_game(written)
    report['replayed_points'] = len(written['points'])
    _write(output, report)
    print(key, report.get('score'), report['metrics'], flush=True)


def verify_resume(folder, mode, fixture, label=None, start=START):
    key = f'{mode}-{fixture_name(fixture, label)}'
    source = path(folder, key)
    saved = path(folder, f'{key}-player')
    receipt = _read(source)
    if _hash(start) != receipt['source_sha256'] or _hash(saved) != receipt['checkpoint_sha256']:
        raise ValueError('checkpoint and initial state must match the recorded game')
    arena = TT8ServiceArena(**receipt['arena'])
    wiring = TT6Wiring(arena, receipt['installation_seed'])
    try:
        cache_hall_stereo(arena, reflections=True)
        _, live = restore(start, arena, mode)
        for expected in receipt['points']:
            row = play_point(live, arena, wiring, receipt['seed'], expected['seed'],
                             expected['phase'], expected['server'])
            if row['action_sha256'] != expected['action_sha256'] or row['evaluator'] != expected['evaluator']:
                raise ValueError(f"RGB policy replay diverged at {expected['seed']}")
        _, resumed = restore(saved, arena, mode)
        seed = 74648933
        continued = []
        for (point_seed, phase), server in zip(schedule(seed, 3249, 4), (-1, 1, -1, 1), strict=True):
            direct = play_point(live, arena, wiring, seed, point_seed, phase, server)
            restored = play_point(resumed, arena, wiring, seed, point_seed, phase, server)
            if direct != restored:
                raise ValueError(f'loaded player diverged at {point_seed}')
            continued.append(dict(seed=point_seed, phase=phase, server=server, action_sha256=direct['action_sha256']))
    finally:
        arena.close()
    _write(path(folder, f'{key}-resume-audit'), dict(game_sha256=_hash(source), checkpoint_sha256=_hash(saved),
                                                    policy_points=len(receipt['points']), continuation=continued))
    print(key, 'verified', len(receipt['points']), 'RGB tapes and', len(continued), 'continuation tapes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    command = parser.add_subparsers(dest='command', required=True)
    play_args = command.add_parser('play')
    play_args.add_argument('mode', choices=PLAY_MODES)
    play_args.add_argument('fixture', type=int, choices=range(len(FIXTURES)))
    play_args.add_argument('--resume', type=Path, default=START)
    play_args.add_argument('--seed', type=int)
    play_args.add_argument('--label')
    audit = command.add_parser('audit')
    audit.add_argument('mode', choices=PLAY_MODES)
    audit.add_argument('fixture', type=int, choices=range(1, len(FIXTURES)))
    audit.add_argument('--resume', type=Path, default=START)
    audit.add_argument('--label')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'play':
        play(args.output, args.mode, args.fixture, start=args.resume, seed=args.seed, label=args.label)
    else:
        verify_resume(args.output, args.mode, args.fixture, args.label, args.resume)
