"""Train first-shot choices, then play frozen controls on the same rotary fixtures."""

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
from .tt16_rotary_player import DEPTHS, FirstShotMemory, FirstShotPlayer, observed_outcome


START = ROOT / 'artifacts/tt15/rotary-blind-game-3-player.json.gz'
SEEDS = dict(practice=190759770, held=798704937, one=2033307149, two=153245035, repeat=90773221)
MODES = (*DEPTHS, 'context', 'global', 'center')
COUNTS = dict(practice=24, held=32, one=56, two=56, repeat=28)


def restore(source, arena, mode, memory):
    state = _read(source)
    cfg = state['rotary_return']
    carry = state['rotary_between_points']
    library, prior, response, timing = load(source)
    player = FirstShotPlayer(library.models[cfg['service_model']], prior, _restore_tracker(arena.cameras, cfg['optics']['commissioned']),
                             response, timing, receiving_model=library.models[cfg['maps']['recalled']],
                             service_belief=cfg['service_belief'], predictive=True,
                             flight_memory=FlightMemory('fixed', carry['flight'], carry['impulses']),
                             service_memory=ServiceMemory(state.get('service_memory')),
                             fixed=int(cfg['profile'].partition('-')[2]),
                             correction=ReturnCorrection(**state['rotary_flight_correction']['correction']),
                             shot_mode=mode, shot_memory=memory)
    prior.rng.bit_generator.state = deepcopy(state['rng'])
    if prior.span_memory is not None:
        span = state['motor_span']
        prior.span_memory.active = span['active']
        prior.span_memory.exposures = {int(k): count for k, count in span['exposures'].items()}
    player.role_rows = [np.asarray(row) for row in carry['roles']]
    return library, player


def read_memory(folder, source):
    state = _read(source)
    if 'first_shot_memory' in state:
        return FirstShotMemory(state['first_shot_memory'])
    fit = path(folder, 'first-fit')
    return FirstShotMemory(_read(fit)['observations'] if fit.exists() else [])


def row_for(player, arena, wiring, seed, point_seed, phase, server, audit_optics=False):
    row = play_point(player, arena, wiring, seed, point_seed, phase, server)
    if audit_optics:
        row['optical_trace'] = [{key: frame[key] for key in ('tick', 'ball', 'estimate', 'horizon', 'goal')}
                                for frame in player.trace]
    return row


def run(folder, fixture, mode, source, seed_override=None):
    state = _read(source)
    if state.get('first_shot_mode', mode) != mode:
        raise ValueError('resume the active first-shot policy stored in the checkpoint')
    if fixture == 'practice' and mode not in (*DEPTHS, 'center'):
        raise ValueError('development practice requires a fixed depth')
    memory = read_memory(folder, source)
    if mode in ('context', 'global') and not memory.observations:
        raise ValueError('fit the first-shot memory before playing a learned policy')
    seed, count = SEEDS[fixture] if seed_override is None else seed_override, COUNTS[fixture]
    held = fixture in ('practice', 'held', 'repeat')
    cases = {'-1': schedule(seed, 4621, count)}
    first_server = 1 if fixture == 'two' else -1
    if not held:
        cases['1'] = schedule(seed, 4622, count)
    config = state['rotary_return']['arena']
    memory_source = source if 'first_shot_memory' in state else path(folder, 'first-fit')
    report = dict(schema='tt16-rotary-first-shot-v1', arena=config, seed=seed, fixture=fixture, mode=mode,
                  installation_seed=config['seed'], source_sha256=_hash(source), schedule=cases,
                  memory_sha256=_hash(memory_source) if mode in ('context', 'global') else None,
                  points=[], practice=held)
    if not held:
        report.update(race_to=11, first_server=first_server)
    arena = TT8ServiceArena(**config)
    wiring = TT6Wiring(arena, config['seed'])
    try:
        cache_hall_stereo(arena, reflections=True)
        library, player = restore(source, arena, mode, memory)
        score = ServiceScore(first_server=first_server)
        used = {'-1': 0, '1': 0}
        for _ in range(count):
            if held:
                _, player = restore(source, arena, mode, memory)
            server = -1 if held else score.server
            side = str(server)
            point_seed, phase = cases[side][used[side]]
            row = row_for(player, arena, wiring, seed, point_seed, phase, server, audit_optics=held)
            report['points'].append(row)
            used[side] += 1
            if not held:
                score.record(row['evaluator'])
                if score.finished:
                    break
        if not held:
            if not score.finished:
                raise RuntimeError(f'{fixture} did not finish')
            saved = path(folder, f'first-{mode}-{fixture}-player')
            checkpoint(saved, source, library, player, 'recalled')
            portable = _read(saved)
            portable.update(rotary_flight_correction=state['rotary_flight_correction'],
                            first_shot_mode=mode, first_shot_memory=memory.observations)
            _write(saved, portable)
            report.update(score=score.score, finished=True, checkpoint_sha256=_hash(saved))
    finally:
        arena.close()
    report['metrics'] = {side: summary([r for r in report['points'] if str(r['server']) == side])
                         for side in cases}
    output = path(folder, f'first-{mode}-{fixture}')
    _write(output, report)
    written = _read(output)
    if held:
        check_tapes(written)
    else:
        verify_game(written)
    report['replayed_points'] = len(written['points'])
    _write(output, report)
    print(mode, fixture, report.get('score'), report['metrics'], flush=True)


def fit(folder):
    observations = []
    schedules = []
    hashes = {}
    for mode in DEPTHS:
        source = path(folder, f'first-{mode}-practice')
        receipt = _read(source)
        hashes[mode] = _hash(source)
        schedules.append(receipt['schedule'])
        for row in receipt['points']:
            shot = row['first_shot']
            if shot is None:
                continue
            optics = observed_outcome(row['optical_trace'], shot['deadline'], row['public_reward'])
            if shot['outcome'] != optics or shot['mode'] != mode:
                raise ValueError('the RGB-only outcome could not be reproduced')
            observations.append(dict(mode=mode, context=shot['context'], outcome=optics))
    if any(schedule != schedules[0] for schedule in schedules[1:]):
        raise ValueError('practice used different incoming trajectories by action')
    memory = FirstShotMemory(observations)
    global_choice, values = memory.choose(np.zeros(4), local=False)
    _write(path(folder, 'first-fit'), dict(observations=observations, sources=hashes,
                                          global_choice=global_choice, values=values))
    print('frozen choice and values', global_choice, values, len(observations), flush=True)


def audit_resume(folder, mode, fixture):
    game = path(folder, f'first-{mode}-{fixture}')
    saved = path(folder, f'first-{mode}-{fixture}-player')
    report = _read(game)
    if _hash(START) != report['source_sha256'] or _hash(saved) != report['checkpoint_sha256']:
        raise ValueError('the game replay requires its archived starting and final checkpoints')
    arena = TT8ServiceArena(**report['arena'])
    wiring = TT6Wiring(arena, report['installation_seed'])
    try:
        cache_hall_stereo(arena, reflections=True)
        _, live = restore(START, arena, mode, read_memory(folder, START))
        for expected in report['points']:
            row = row_for(live, arena, wiring, report['seed'], expected['seed'], expected['phase'], expected['server'])
            if row['action_sha256'] != expected['action_sha256'] or row['first_shot'] != expected['first_shot']:
                raise ValueError(f"RGB policy replay failed at {expected['seed']}")
        _, resumed = restore(saved, arena, mode, read_memory(folder, saved))
        seed = 17282055
        continuation = []
        for (point_seed, phase), server in zip(schedule(seed, 3249, 4), (-1, 1, -1, 1), strict=True):
            direct = row_for(live, arena, wiring, seed, point_seed, phase, server)
            restored = row_for(resumed, arena, wiring, seed, point_seed, phase, server)
            if direct != restored:
                raise ValueError(f'continued state diverged at {point_seed}')
            continuation.append(dict(seed=point_seed, phase=phase, server=server, action_sha256=direct['action_sha256']))
    finally:
        arena.close()
    _write(path(folder, f'first-{mode}-{fixture}-audit'), dict(game_sha256=_hash(game), checkpoint_sha256=_hash(saved),
                                                             replayed=len(report['points']), continuation=continuation))
    print(mode, fixture, 'verified RGB game and continuation', len(report['points']), len(continuation), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    cmd = parser.add_subparsers(dest='command', required=True)
    run_args = cmd.add_parser('play')
    run_args.add_argument('fixture', choices=SEEDS)
    run_args.add_argument('mode', choices=MODES)
    run_args.add_argument('--resume', type=Path, default=START)
    run_args.add_argument('--seed', type=int)
    cmd.add_parser('fit')
    audit_args = cmd.add_parser('audit')
    audit_args.add_argument('mode', choices=MODES)
    audit_args.add_argument('fixture', choices=('one', 'two'))
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    if args.command == 'play':
        run(args.folder, args.fixture, args.mode, args.resume, args.seed)
    elif args.command == 'fit':
        fit(args.folder)
    else:
        audit_resume(args.folder, args.mode, args.fixture)
