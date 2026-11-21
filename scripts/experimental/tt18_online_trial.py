"""Compare body priors through consecutive games and replay their scored tapes."""

import argparse
import json
from hashlib import sha256
from itertools import islice
from pathlib import Path

import numpy as np

from .tt4_camera import PerspectiveCamera
from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
from .tt6_continuous_trial import match_phases, prepare_point
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import ServiceScore
from .tt8_service_trial import PointInterface
from .tt10_series_trial import PHASES, _game_summary, _restore, _restore_tracker, _save
from .tt10_window_trial import _point_row
from .tt14_gain_hardware import perturbation
from .tt16_transfer_game import ROW_FIELDS
from .tt17_motor_protocol import make_arena as make_tt17_arena, wiring_for
from .tt18_online import (BodyLearning, OnlineBodyPlayer, OnlineServePlayer, RecallLearning, ServeChoices,
                          blank_body, carry_body)


MODES = ('carried', 'cold', 'passive', 'frozen', 'serve', 'recall')
SITES = {'gantry': 4, 'five': 5}


def donor_for(site):
    return ROOT / 'artifacts/tt17' / f'motor-{site}-coupling-bounded-start-player.json.gz'


def fixture(site, plant, seed, *, fresh_ports=False):
    path = donor_for(site)
    donor = _read(path)
    arena = donor['series']['arena']
    mixer, ports = np.random.default_rng([seed, 754283]).integers(0, 2**31, size=2).tolist()
    return dict(arena=arena, camera=donor['series']['tt16_camera'], plant=plant,
                wiring=ports if fresh_ports else arena['seed'],
                matrix=perturbation(SITES[site], mixer, 'coupling').tolist())


def new_arena(spec):
    if spec['plant'] is None:
        return make_tt17_arena(spec)
    from .tt18_plant import make_arena

    return make_arena(spec)


def startup(path, mode, seed):
    saved_series = _read(path)['series']
    camera = tuple(PerspectiveCamera(**eye) for eye in saved_series['tt16_camera'])
    tracker = _restore_tracker(camera, saved_series['optics'])
    state, library, player = _restore(path, tracker_factory=tracker)
    player.__class__ = OnlineServePlayer if mode == 'serve' else OnlineBodyPlayer
    learner = RecallLearning if mode == 'recall' else BodyLearning
    if 'tt18_online' in state:
        if state['tt18_online']['mode'] != mode:
            raise ValueError('requested mode differs from the checkpoint')
        player.body_learning = learner.restore(state['tt18_online']['learning'])
    else:
        active = state['gain_memory']['maps'][state['gain_memory']['active']]
        base = library.models[active['model']]
        prior = blank_body(base.action_size) if mode == 'cold' else carry_body(base, np.asarray(active['transform']))
        player.body_learning = learner(prior, enabled=mode != 'frozen', seed=seed, directed=mode != 'passive')
    player.adapter = player.body_learning.model
    if mode == 'serve':
        player.serve_choices = ServeChoices(state.get('tt18_online', {}).get('serve_choices', ()))
    if player.prior.span_memory is not None:
        player.prior.span_memory.active = None
    return state, library, player


def save_checkpoint(path, spec, state, library, player, mode):
    model = player.body_learning.model
    library.models.append(model)
    old = state['series']
    entry = {k: v for k, v in old.items() if k not in ('completed', 'role_rows', 'flight_history', 'flight_impulses')}
    entry.update(model=len(library.models) - 1, arena=spec['arena'], tt16_camera=spec['camera'])
    _save(path, state, library, player, entry, old['completed'] + 1)
    portable = _read(path)
    portable['tt18_online'] = dict(mode=mode, learning=player.body_learning.state())
    if mode == 'serve':
        portable['tt18_online']['serve_choices'] = player.serve_choices.observations
    _write(path, portable)


def schedule(seed):
    starts = _seeds(seed, 6011, 0, 65)['tests']
    phases = match_phases(PHASES, 65, [seed, 6011])
    return zip(starts, phases, strict=True)


def play_point(player, arena, wires, game_seed, point, phase, server):
    learner = player.body_learning
    before = learner.updates
    raw = _point_row(player, arena, wires, game_seed, point, phase, server)
    row = {field: raw[field] for field in ROW_FIELDS}
    row['optical'] = [[frame[key] for key in ('ball', 'blade', 'blade_pitch', 'goal', 'horizon',
                                              'body_prediction_error', 'dither')] for frame in player.trace]
    row['body'] = dict(updates_before=before, updates_after=learner.updates,
                       sightings=learner.sightings, uncertainty=learner.uncertainty.tolist())
    if 'serve_exploration' in raw:
        row['serve_exploration'] = raw['serve_exploration']
    if isinstance(learner, RecallLearning):
        learner.end_point()
        row['basis_recall'] = learner.decision
    return row


def metrics(rows):
    result = _game_summary(rows)
    result['frames'] = sum(row['steps'] for row in rows)
    result['serve_contacts'] = sum(row['evaluator']['service_contact'] for row in rows if row['server'] == 1)
    errors = [frame[5] for row in rows for frame in row['optical'] if frame[5] is not None]
    result['body_prequential_error_median_m'] = float(np.median(errors)) if errors else None
    result['body_sightings'] = len(errors)
    result['exploration_rate_frames'] = sum(frame[6] for row in rows for frame in row['optical'][:-1])
    first = {'legal_serve': None, 'legal_return': None, 'second_legal_return': None}
    elapsed = 0
    for row in rows:
        returns = 0
        for event in row['events']:
            if event.get('side') != 1:
                continue
            kind = event['kind']
            if kind == 'legal_return':
                returns += 1
                if returns == 2:
                    kind = 'second_legal_return'
            if kind in first and first[kind] is None:
                first[kind] = elapsed + event['step']
        elapsed += row['steps']
    result['first_frames'] = first
    return result


def run(folder, site, plant, seed, mode, games, *, fresh_ports=False):
    if games < 1:
        raise ValueError('expected at least one game')
    spec = fixture(site, plant, seed, fresh_ports=fresh_ports)
    layout = '-newports' if fresh_ports else ''
    stem = f'online-{site}-{plant or "classic"}{layout}-{seed}-{mode}'
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / f'{stem}.json.gz'
    donor = donor_for(site)
    arena = new_arena(spec)
    seeds = _seeds(seed, 9387, 0, games)['tests']
    if len(set(seeds)) != len(seeds):
        raise ValueError('game seeds must be distinct')
    report = dict(schema='tt18-online-body-v1', site=site, mode=mode, fixture=spec, stream_seed=seed,
                  donor_sha256=_hash(donor), target_preparation_frames=0, fresh_ports=fresh_ports,
                  control_dt=arena.control_dt, games=[],
                  optical_fields=['ball', 'blade', 'blade_pitch', 'goal', 'horizon', 'body_prediction_error_m',
                                  'exploration_rate_delta'])
    try:
        cache_hall_stereo(arena, reflections=True)
        wires = wiring_for(arena, spec)
        state, library, player = startup(donor, mode, seed)
        for index, game_seed in enumerate(seeds, 1):
            score = ServiceScore(first_server=-1)
            rows = []
            for point, phase in schedule(game_seed):
                row = play_point(player, arena, wires, game_seed, point, phase, score.server)
                rows.append(row)
                score.record(row['evaluator'])
                if score.finished:
                    break
            if not score.finished:
                raise RuntimeError('the scored game exhausted its point schedule')
            checkpoint = folder / f'{stem}-game{index}-player.json.gz'
            save_checkpoint(checkpoint, spec, state, library, player, mode)
            game = dict(seed=game_seed, score=list(score.score), points=rows, metrics=metrics(rows),
                        checkpoint=checkpoint.name, checkpoint_sha256=_hash(checkpoint))
            report['games'].append(game)
            _write(receipt, report)
            if index < games:
                state, library, player = startup(checkpoint, mode, seed)
            print(stem, index, score.score, {key: game['metrics'][key] for key in (
                'legal_own_serves', 'own_serves', 'total_legal_returns', 'two_return_points',
                'body_prequential_error_median_m', 'body_sightings')}, flush=True)
    finally:
        arena.close()
    report['replayed_physics_points'] = verify(receipt)
    _write(receipt, report)
    return receipt


def verify(receipt, *, rgb=False):
    report = _read(receipt)
    spec = report['fixture']
    arena = new_arena(spec)
    donor = donor_for(report['site'])
    total = 0
    try:
        wires = wiring_for(arena, spec)
        if _hash(donor) != report['donor_sha256']:
            raise ValueError('player donor digest changed')
        if rgb:
            cache_hall_stereo(arena, reflections=True)
            _, _, player = startup(donor, report['mode'], report['stream_seed'])
        seen = set()
        for game in report['games']:
            score = ServiceScore(first_server=-1)
            starts = islice(schedule(game['seed']), len(game['points']))
            for row, (seed, phase) in zip(game['points'], starts, strict=True):
                if (score.finished or seed in seen
                        or (seed, phase, score.server) != (row['seed'], row['phase'], row['server'])):
                    raise ValueError('game schedule or service order changed')
                seen.add(seed)
                if len(row['actions']) != row['steps'] or len(row['optical']) != row['steps'] + 1:
                    raise ValueError('control frames or stereo sightings are missing from the point tape')
                if rgb:
                    actual = play_point(player, arena, wires, game['seed'], seed, phase, score.server)
                    if json.loads(json.dumps(actual)) != row:
                        raise ValueError(f'optical replay changed on scored point {seed}')
                else:
                    interface = PointInterface(wires, prepare_point(arena, phase), server=score.server,
                                               seed=game['seed'])
                    interface.reset(seed=seed, render=False)
                    for action in row['actions']:
                        interface.step(action, render=False)
                    digest = sha256(np.asarray(row['actions'], dtype='<f8').tobytes()).hexdigest()
                    if (arena.point_result() != row['evaluator'] or interface.applied != row['applied']
                            or json.loads(json.dumps(arena.events)) != row['events']
                            or digest != row['action_sha256']):
                        raise ValueError(f'physical replay changed on scored point {seed}')
                score.record(row['evaluator'])
                total += 1
            path = receipt.parent / game['checkpoint']
            if _hash(path) != game['checkpoint_sha256']:
                raise ValueError('between-game player checkpoint changed')
            if not score.finished or list(score.score) != game['score'] or metrics(game['points']) != game['metrics']:
                raise ValueError('saved game score or metrics changed')
            if rgb:
                saved = _read(path)['tt18_online']
                if player.body_learning.state() != saved['learning']:
                    raise ValueError('body model did not replay through the scored optical observations')
                if 'serve_choices' in saved and player.serve_choices.observations != saved['serve_choices']:
                    raise ValueError('service variation choices did not replay')
                _, _, player = startup(path, report['mode'], report['stream_seed'])
    finally:
        arena.close()
    return total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--site', choices=SITES)
    parser.add_argument('--plant', choices=('compact', 'extended'))
    parser.add_argument('--mode', choices=MODES)
    parser.add_argument('--seed', type=int, default=85729317)
    parser.add_argument('--games', type=int, default=2)
    parser.add_argument('--fresh-ports', action='store_true')
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--rgb', action='store_true')
    args = parser.parse_args()
    if args.verify:
        count = verify(args.verify, rgb=args.rgb)
        if args.rgb:
            report = _read(args.verify)
            report['replayed_rgb_points'] = count
            _write(args.verify, report)
        print('replayed', count, flush=True)
    elif args.site is None or args.mode is None:
        parser.error('provide --site and --mode for scored play')
    else:
        run(args.folder, args.site, args.plant, args.seed, args.mode, args.games, fresh_ports=args.fresh_ports)
