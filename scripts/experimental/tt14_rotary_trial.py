"""Compare visually learned forward and upward receiving corrections on rotary."""

import argparse
from copy import deepcopy
from dataclasses import asdict
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
from .tt12_body_trial import own_service_prefix, play_point, summary, verify_game
from .tt13_rotary_report import paired_returns, same_incoming_events
from .tt13_rotary_trial import add_diagnostics, checkpoint, path, restore as restore_baseline, schedule
from .tt14_rotary_player import FlightCorrectedPlayer, ReturnCorrection, learn


START = ROOT / 'artifacts/tt13/rotary-recalled-game-2-player.json.gz'
TRAIN = ROOT / 'artifacts/tt13/rotary-recalled-held.json.gz'
HELD_SEED = 802520353
FIXTURES = (1777891, 93457782)
MODES = ('baseline', 'pace', 'loft', 'recenter')
PLAY_MODES = (*MODES, 'pace-relative')


def train(folder):
    rows = _read(TRAIN)['points']
    traces = [(row['optical_trace'], row['optical']['contact_prediction']) for row in rows]
    correction, samples = learn(traces)
    receipt = dict(schema='tt14-rotary-flight-fit-v1', training_sha256=_hash(TRAIN),
                   start_sha256=_hash(START), starts=len(traces), samples=samples,
                   correction=asdict(correction))
    _write(path(folder, 'flight-fit'), receipt)
    print('optical return correction', correction, 'sightings', len(samples), flush=True)


def restore(start, arena, mode, correction):
    if mode == 'baseline':
        return restore_baseline(start, arena, 'recalled')
    state = _read(start)
    cfg = state['rotary_return']
    memory = state['rotary_between_points']
    library, prior, response, timing = load(start)
    service = library.models[cfg['service_model']]
    receiving = library.models[cfg['maps']['recalled']]
    optics = _restore_tracker(arena.cameras, cfg['optics']['commissioned'])
    flight = FlightMemory('fixed', memory['flight'], memory['impulses'])
    player = FlightCorrectedPlayer(service, prior, optics, response, timing,
                                   receiving_model=receiving, service_belief=cfg['service_belief'],
                                   flight_memory=flight, predictive=True, correction=correction, mode=mode,
                                   service_memory=ServiceMemory(state.get('service_memory')),
                                   fixed=int(cfg['profile'].partition('-')[2]))
    prior.rng.bit_generator.state = deepcopy(state['rng'])
    if prior.span_memory is not None:
        span = state['motor_span']
        prior.span_memory.active = span['active']
        prior.span_memory.exposures = {int(key): count for key, count in span['exposures'].items()}
    player.role_rows = [np.asarray(row) for row in memory['roles']]
    return library, player


def play(folder, mode, fixture, *, start=START, seed=None, label=None):
    fit_path = path(folder, 'flight-fit')
    correction = ReturnCorrection(**_read(fit_path)['correction'])
    state = _read(start)
    cfg = state['rotary_return']
    saved = state.get('rotary_flight_correction')
    if saved is not None and (saved['mode'] != mode or saved['correction'] != asdict(correction)):
        raise ValueError('resume the saved flight correction in its original mode')
    held = fixture == 0
    if seed is None:
        seed = HELD_SEED if held else FIXTURES[fixture - 1]
    cases = {'-1': schedule(seed, 5733, 24 if held else 56)}
    first_server = 1 if fixture == 2 else -1
    if not held:
        cases['1'] = schedule(seed, 5734, 56)
    key = f'{mode}-held' if held else f'{mode}-game-{fixture}'
    if label:
        key = f'{key}-{label}'
    cfg_arena = cfg['arena']
    report = dict(schema='tt14-rotary-flight-v1', arena=cfg_arena, seed=seed,
                  installation_seed=cfg_arena['seed'], mode=mode, fixture=fixture,
                  start_sha256=_hash(start), flight_fit_sha256=_hash(fit_path), schedule=cases,
                  points=[], practice=held)
    if not held:
        report.update(race_to=11, first_server=first_server)
    arena = TT8ServiceArena(**cfg_arena)
    wiring = TT6Wiring(arena, cfg_arena['seed'])
    try:
        cache_hall_stereo(arena, reflections=True)
        library, player = restore(start, arena, mode, correction)
        score = ServiceScore(first_server=first_server)
        used = {'-1': 0, '1': 0}
        for _ in range(24 if held else 56):
            if held:
                _, player = restore(start, arena, mode, correction)
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
            destination = path(folder, f'{key}-player')
            checkpoint(destination, start, library, player, 'recalled')
            portable = _read(destination)
            portable['rotary_flight_correction'] = dict(mode=mode, correction=asdict(correction))
            _write(destination, portable)
            report.update(score=score.score, finished=True, checkpoint_sha256=_hash(destination))
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


def audit_resume(folder, mode, fixture):
    receipt_path = path(folder, f'{mode}-game-{fixture}')
    saved = path(folder, f'{mode}-game-{fixture}-player')
    receipt = _read(receipt_path)
    if (_hash(START) != receipt['start_sha256'] or _hash(saved) != receipt['checkpoint_sha256']):
        raise ValueError('audit expects a game started from the canonical checkpoint')
    correction = ReturnCorrection(**_read(path(folder, 'flight-fit'))['correction'])
    arena = TT8ServiceArena(**receipt['arena'])
    wiring = TT6Wiring(arena, receipt['installation_seed'])
    try:
        cache_hall_stereo(arena, reflections=True)
        _, live = restore(START, arena, mode, correction)
        for expected in receipt['points']:
            row = play_point(live, arena, wiring, receipt['seed'], expected['seed'],
                             expected['phase'], expected['server'])
            if row['action_sha256'] != expected['action_sha256'] or row['evaluator'] != expected['evaluator']:
                raise ValueError(f"RGB game replay differed at {expected['seed']}")
        _, restored = restore(saved, arena, mode, correction)
        seed = 874305612
        continuations = []
        for (point_seed, phase), server in zip(schedule(seed, 1702, 4), (-1, 1, -1, 1), strict=True):
            direct = play_point(live, arena, wiring, seed, point_seed, phase, server)
            loaded = play_point(restored, arena, wiring, seed, point_seed, phase, server)
            if direct != loaded:
                raise ValueError(f'checkpoint continuation differed at {point_seed}')
            continuations.append(dict(seed=point_seed, phase=phase, server=server,
                                      action_sha256=direct['action_sha256'], reward=direct['public_reward']))
    finally:
        arena.close()
    _write(path(folder, f'{mode}-resume-audit'), dict(game_sha256=_hash(receipt_path), checkpoint_sha256=_hash(saved),
                                                     policy_points=len(receipt['points']), seed=seed,
                                                     continuation=continuations))
    print(mode, 'RGB replay', len(receipt['points']), 'continuation', len(continuations), flush=True)


def matched_control(baseline, alternative):
    result = paired_returns(baseline, alternative)
    before, after = ({row['seed']: own_service_prefix(row) for row in receipt['points'] if row['server'] == 1}
                     for receipt in (baseline, alternative))
    pairs = [(before[seed], after[seed]) for seed in before.keys() & after.keys()]
    result['identical_service_commands'] = sum(a[:2] == b[:2] for a, b in pairs)
    result['equivalent_service_events'] = sum(a[2] == b[2] or same_incoming_events(a[2], b[2]) for a, b in pairs)
    return result


def compare(folder, modes, label=None):
    result = {}
    for fixture in ('held', 'game-1', 'game-2'):
        key = f'{fixture}-{label}' if label else fixture
        records = {mode: _read(path(folder, f'{mode}-{key}')) for mode in modes}
        baseline = records['baseline']
        result[fixture] = dict(branches={mode: {k: r.get(k) for k in ('metrics', 'score', 'replayed_points')}
                                         for mode, r in records.items()},
                               pairs={mode: matched_control(baseline, r) for mode, r in records.items()
                                      if mode != 'baseline'})
    output = f'comparison-{label}' if label else 'comparison'
    _write(path(folder, output), result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    command = parser.add_subparsers(dest='command', required=True)
    command.add_parser('train')
    play_parser = command.add_parser('play')
    play_parser.add_argument('mode', choices=PLAY_MODES)
    play_parser.add_argument('fixture', type=int, choices=(0, 1, 2))
    play_parser.add_argument('--resume', type=Path, default=START)
    play_parser.add_argument('--seed', type=int)
    play_parser.add_argument('--label')
    audit_parser = command.add_parser('audit-resume')
    audit_parser.add_argument('mode', choices=PLAY_MODES)
    audit_parser.add_argument('fixture', type=int, choices=(1, 2))
    comparison = command.add_parser('compare')
    comparison.add_argument('modes', nargs='*', default=MODES)
    comparison.add_argument('--label')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'train':
        train(args.output)
    elif args.command == 'play':
        play(args.output, args.mode, args.fixture, start=args.resume, seed=args.seed, label=args.label)
    elif args.command == 'audit-resume':
        audit_resume(args.output, args.mode, args.fixture)
    else:
        compare(args.output, args.modes, args.label)
