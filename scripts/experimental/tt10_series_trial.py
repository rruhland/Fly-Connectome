"""Compare continuing and restarting the ordinary registered player across full games."""
import argparse
from base64 import b64decode, b64encode
from copy import deepcopy
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
from .tt5_vision import HallTracker, _Stereo
from .tt6_body_arena import TT6Wiring
from .tt6_body_vision import BodyTracker
from .tt6_continuous_trial import match_phases
from .tt6_memory import load
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring
from .tt7_flight import FlightMemory
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_trial import check_tapes
from .tt9_precontact_match import save_player
from .tt10_window_trial import _player, _point_row, _summary


SOURCE = ROOT / 'artifacts/tt9/precontact-match-b-player.json.gz'
PHASES = ('ordinary', 'mixed', 'soft', 'delay')


def _capture_tracker(factory):
    tracker = factory()
    reference = tracker.reference
    return dict(tracker=type(tracker).__name__, shape=reference.shape, dtype=reference.dtype.str,
                background=b64encode(reference.tobytes()).decode('ascii'),
                apertures=[[(int(axis.start), int(axis.stop)) for axis in pair] for pair in tracker.apertures],
                motion_supported_blade=tracker.motion_supported_blade)


def _restore_tracker(rig, state):
    tracker = {'BodyTracker': BodyTracker, 'HallTracker': HallTracker}[state['tracker']]
    reference = np.frombuffer(b64decode(state['background']), dtype=state['dtype']).reshape(state['shape'])
    stereo = _Stereo(rig)
    apertures = tuple(tuple(slice(*axis) for axis in pair) for pair in state['apertures'])
    return lambda: tracker(stereo, reference, apertures, motion_supported_blade=state['motion_supported_blade'])


def _save(path, previous, library, player, entry, completed):
    save_player(path, previous, library, player)
    portable = _read(path)
    portable['series'] = dict(**entry, completed=completed,
                              role_rows=np.asarray(player.role_rows).tolist(),
                              flight_history=list(player.memory.history), flight_impulses=list(player.memory.impulses))
    _write(path, portable)


def _restore(path, arena=None, *, tracker_factory=None):
    state = _read(path)
    entry = state['series']
    library, prior, response, timing = load(path)
    fit = library.models[entry['model']]
    factory = (tracker_factory if tracker_factory is not None
               else _restore_tracker(arena.cameras, entry['optics']))
    player = _player('fixed', fit, factory, prior, response, timing, state)
    # The constructor initializes/reset-samples a new player. Restore the exact
    # between-point choice stream and existing installation assignment.
    player.prior.rng.bit_generator.state = deepcopy(state['rng'])
    if player.prior.span_memory is not None:
        memory = player.prior.span_memory
        memory.active = state['motor_span']['active']
        memory.exposures = {int(k): n for k, n in state['motor_span']['exposures'].items()}
    player.role_rows = [np.asarray(row) for row in entry['role_rows']]
    player.memory = FlightMemory('fixed', entry['flight_history'], entry['flight_impulses'])
    return state, library, player


def prepare(args):
    library, prior, response, timing = load(args.donor)
    state = _read(args.donor)
    install = _seeds(args.installation_seed, 0, 1, 1)
    arena_args = dict(seed=install['arena_seed'], view_seed=install['view_seed'], body=args.body, venue='hall',
                      view='first_person', render_mode='mujoco', width=192, height=144, baseline=.28,
                      max_steps=420, frameskip=8, distractor=False)
    arena = TT8ServiceArena(**arena_args)
    wiring = TT6Wiring(arena, arena_args['seed'])
    schedule = _seeds(args.calibration_seed, 0, 8, 0)
    try:
        cache_hall_stereo(arena, reflections=True)
        fit, factory, probes, attempts = library.commission(
            DelayedWiring(wiring), arena.cameras, schedule, probe_cache=args.probe_cache)
        if fit is None:
            raise ValueError(f'body identification failed: {attempts}')
        model = next(index for index, candidate in enumerate(library.models) if candidate is fit)
        player = _player('fixed', fit, factory, prior, response, timing, state)
        entry = dict(arena=arena_args, model=model, optics=_capture_tracker(factory),
                     calibration=schedule, probes=probes, attempts=attempts, donor_sha256=_hash(args.donor))
        _save(args.output, state, library, player, entry, 0)
    finally:
        arena.close()
    print(f'commissioned in {len(probes)} probe episodes', flush=True)


def _game_summary(rows):
    return dict(_summary(rows),
                sustained_points=sum(row['summary']['sustained'] for row in rows),
                two_return_points=sum(row['player_returns'] >= 2 for row in rows),
                wins_on_own_serve=sum(row['points_for'] for row in rows if row['server'] == 1))


def verify(receipt):
    total = 0
    for game in receipt['games']:
        if game['metrics'] != _game_summary(game['points']) or not game['finished']:
            raise ValueError(f"incomplete or inconsistent game {game['seed']}")
        check_tapes(dict(receipt, **game))
        total += len(game['points'])
    return total


def compare(carry, fresh):
    for key in ('source_sha256', 'arena', 'series_seed', 'game_seeds'):
        if carry[key] != fresh[key]:
            raise ValueError(f'series differ in {key}')
    if carry['mode'] != 'carry' or fresh['mode'] != 'fresh':
        raise ValueError('comparison requires carried and restarted series')
    if not (len(carry['games']) == len(fresh['games']) == len(carry['game_seeds'])):
        raise ValueError('series must finish all scheduled games')
    if carry['games'][0] != fresh['games'][0]:
        raise ValueError('the first match differed under identical initial state')
    return [dict(seed=a['seed'], carry=a['metrics'], fresh=b['metrics'],
                 carry_score=a['score'], fresh_score=b['score'])
            for a, b in zip(carry['games'], fresh['games'], strict=True)]


def run(args):
    source = _read(args.resume)['series']
    source_sha256 = _hash(args.resume)
    arena = TT8ServiceArena(**source['arena'])
    wiring = TT6Wiring(arena, source['arena']['seed'])
    game_seeds = _seeds(args.seed, 9387, 0, args.games)['tests']
    report = dict(schema='tt10-registered-series-v1', mode=args.mode, arena=source['arena'],
                  installation_seed=source['arena']['seed'], series_seed=args.seed,
                  game_seeds=game_seeds, source_sha256=source_sha256, game_commissions=0, games=[],
                  implementation_sha256=_hash(Path(__file__)))
    entry = {key: source[key] for key in ('arena', 'model', 'optics', 'calibration', 'probes', 'attempts', 'donor_sha256')}
    try:
        cache_hall_stereo(arena, reflections=True)
        state, library, player = _restore(args.resume, arena)
        for index, seed in enumerate(game_seeds):
            if args.mode == 'fresh' and index:
                _, _, player = _restore(args.resume, arena)
            score = ServiceScore(first_server=-1)
            starts = _seeds(seed, 6011, 0, args.limit)['tests']
            phases = match_phases(PHASES, args.limit, [seed, 6011])
            game = dict(seed=seed, race_to=11, first_server=-1, points=[], schedule=list(zip(starts, phases)))
            report['games'].append(game)
            for point_seed, phase in game['schedule']:
                row = _point_row(player, arena, wiring, seed, point_seed, phase, score.server)
                game['points'].append(row)
                score.record(row['evaluator'])
                if score.finished:
                    break
            if not score.finished:
                raise RuntimeError(f'game {index + 1} exceeded the point limit')
            game.update(score=score.score, finished=True, metrics=_game_summary(game['points']))
            if _hash(args.resume) != source_sha256:
                raise RuntimeError('source checkpoint was mutated')
            if args.checkpoint:
                _save(args.checkpoint, state, library, player, entry, source['completed'] + index + 1)
            _write(args.output, report)
            print(index + 1, seed, score.score, game['metrics'], flush=True)
    finally:
        arena.close()
    print(f"{verify(_read(args.output))} action tapes replayed", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--donor', type=Path, default=SOURCE)
    p.add_argument('--body', choices=('offset', 'gantry'), default='offset')
    p.add_argument('--installation-seed', type=int, default=317557501)
    p.add_argument('--calibration-seed', type=int, default=926138271)
    p.add_argument('--probe-cache', type=Path)
    p.add_argument('--seed', type=int)
    p.add_argument('--mode', choices=('carry', 'fresh'), default='carry')
    p.add_argument('--games', type=int, default=5)
    p.add_argument('--limit', type=int, default=65)
    p.add_argument('--resume', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--compare', nargs=2, type=Path)
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    if args.prepare:
        prepare(args)
    elif args.verify:
        print(f'{verify(_read(args.output))} action tapes replayed')
    elif args.compare:
        result = compare(*map(_read, args.compare))
        _write(args.output, dict(schema='tt10-registered-series-comparison-v1', games=result))
        print(result)
    elif (args.seed is None or args.resume is None or args.games < 1 or args.limit < 1 or
          args.resume.resolve() in {path.resolve() for path in (args.output, args.checkpoint) if path is not None}):
        p.error('supply a seed, positive limits, and distinct input and output paths')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        run(args)
