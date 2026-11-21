"""Run matched optical SO-101 games with charged setup and replayable checkpoints."""

import argparse
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import match_phases, prepare_point
from .tt6_memory import load
from .tt7_flight import FlightMemory
from .tt8_service_arena import ServiceScore
from .tt8_service_trial import PointInterface
from .tt10_series_trial import _game_summary
from .tt10_window_trial import _point_row
from .tt16_transfer_game import ROW_FIELDS
from .tt19_flight import FlightLaw
from .tt19_online import Court, OpticalPlayer
from .tt19_online_body import InverseModel, blade_pose, kernel_for
from .tt19_probe import command_tape
from .tt19_view import ViewTransport
from .tt19_view_trial import SO101ViewArena


DONOR = ROOT / 'artifacts/tt17/motor-gantry-coupling-travel-player.json.gz'
SOURCE_FLIGHT = ROOT / 'artifacts/tt19/flight-source.json.gz'
DEFAULT_SEED = 439071433
MODES = ('warm', 'cold', 'pose', 'pose-cold', 'frozen', 'nominal-flight')
PHASES = ('ordinary', 'ordinary', 'soft', 'mixed')


class SO101Wiring(TT6Wiring):
    """Permute the motion ports while preserving the public gripper channel."""

    def __init__(self, arena, seed):
        self.arena = arena
        rng = np.random.default_rng([seed, 24889])
        wrist = arena.action_size - 1
        self.order = np.r_[rng.permutation(wrist), wrist]
        self.signs = np.r_[rng.choice((-1, 1), wrist), 1]


def specification(seed, view, strict=False):
    settings = _seeds(seed, 71709, 1, 0)
    arena = dict(seed=settings['arena_seed'], view_seed=settings['view_seed'], view_mode=view,
                 venue='hall', view='first_person', render_mode='mujoco', frameskip=8,
                 max_steps=320, distractor=False, detail='preview')
    if strict:
        arena['strict_endline'] = True
    return dict(arena=arena,
                court=dict(half_length=.80, half_width=.424, net_clearance=.1725),
                public_frame_dt=1 / 30, layout_seed=settings['player_seed'], setup_seed=settings['calibration'][0])


def startup(path, width, court, view, mode, seed, solver='bounded'):
    state = _read(path)
    stored = state.get('tt19_online')
    if stored is not None:
        if stored['mode'] != mode:
            raise ValueError('requested controller differs from the saved mode')
        transport = ViewTransport(view, stored['views'])
        body = InverseModel(width, stored['motor']['kernel'], stored['motor']['mode'], seed, state=stored['motor'])
        law = FlightLaw.from_state(stored['law'])
        response = FlightMemory('impulse', stored['flight_history'], stored['flight_impulses'])
    else:
        transport = ViewTransport(view)
        archive = state['gain_memory']
        index = archive['maps'][archive['active']]['model']
        cold = mode in ('cold', 'pose-cold')
        old = None if cold else load(path)[0].models[index]
        body = InverseModel(width, kernel_for(old), 'pose' if mode.startswith('pose') else 'secant', seed,
                            solver=solver)
        law = (FlightLaw.from_state(_read(SOURCE_FLIGHT)['source_law'])
               if not cold and mode != 'nominal-flight' else FlightLaw())
        prior = {} if cold else state['series']
        response = FlightMemory('impulse', prior.get('flight_history', ()), prior.get('flight_impulses', ()))
    player = OpticalPlayer(body, transport.tracker, court, law, response)
    if stored is not None:
        player.bounce_samples = list(stored['bounce_samples'])
        player.shot_samples = stored['shot_samples']
        player.explore = stored['explore']
        player.registration.update(stored.get('pitch_registration', {}))
        player.registration_count.update(stored.get('pitch_updates', {}))
    return state, player, transport


def save_checkpoint(path, old, player, transport, mode, origin):
    history = list(player.response.history)
    impulses = list(player.response.impulses)
    prior_bounces = len(old.get('tt19_online', {}).get('bounce_samples', ()))
    current = dict(old, bounce=(old['bounce'] + player.bounce_samples[prior_bounces:])[-40:],
                   series=dict(old['series'], flight_history=history, flight_impulses=impulses))
    source_world = old.get('tt19_online', {}).get('gantry_world')
    if source_world is None and 'tt19_online' not in old:
        source_world = dict(bounce=old['bounce'], flight_history=old['series']['flight_history'],
                            flight_impulses=old['series']['flight_impulses'])
    current['tt19_online'] = dict(mode=mode, donor_sha256=origin, motor=player.body.state(), law=player.law.state(),
                                  views=transport.state(), bounce_samples=player.bounce_samples,
                                  shot_samples=player.shot_samples, explore=player.explore,
                                  flight_history=history, flight_impulses=impulses)
    if source_world is not None:
        current['tt19_online']['gantry_world'] = source_world
    if player.body.solver == 'register':
        current['tt19_online'].update(pitch_registration=player.registration, pitch_updates=player.registration_count)
    _write(path, current)


def schedule(seed):
    starts = _seeds(seed, 28615, 0, 65)['tests']
    phases = match_phases(PHASES, 65, [seed, 28615])
    return zip(starts, phases, strict=True)


def calibration(player, arena, wiring, game_seed, seed, count):
    if count == 0:
        return dict(frames=0, actions=[], views=[])
    interface = PointInterface(wiring, prepare_point(arena, 'ordinary'), server=-1, seed=game_seed)
    tracker = player.tracker_factory()
    model = player.body
    model.reset()
    images = interface.reset(seed=seed)
    actions, views = [], []
    tracker.update(images)
    observed = blade_pose(tracker)
    model.measure(observed)
    views.append(None if observed is None else finite_pose(observed))
    for action in command_tape(model.width, count, kernel=model.kernel):
        action = action.copy()
        action[-1] = 0.
        images, _, done, time_up, _ = interface.step(action)
        actions.append(action.tolist())
        model.advance(action)
        tracker.update(images)
        observed = blade_pose(tracker)
        model.measure(observed)
        views.append(None if observed is None else finite_pose(observed))
        if done or time_up:
            break
    return dict(seed=seed, frames=len(actions), allotted=count, actions=actions, views=views,
                observed=sum(row is not None for row in views), applied=interface.applied)


def finite_pose(pose):
    return [float(value) if np.isfinite(value) else None for value in pose]


def play(folder, *, seed=DEFAULT_SEED, mode='warm', view='wrist+fixed', games=2, setup_frames=0,
         limit=None, resume=None, solver='bounded', game_offset=0, strict=False):
    if games < 1 or game_offset < 0 or (resume is not None and game_offset < 1):
        raise ValueError('a continued stream requires its next unused game offset')
    spec = specification(seed, view, strict)
    court = Court(**spec['court'])
    arena = SO101ViewArena(**spec['arena'])
    folder.mkdir(parents=True, exist_ok=True)
    prefix = f'online-{seed}-{mode}-{view}' + (f'-setup{setup_frames}' if setup_frames else '')
    if strict:
        prefix += '-strict'
    if solver == 'register':
        prefix += '-register'
    if game_offset:
        prefix += f'-continued{game_offset}'
    if limit is not None:
        prefix += '-pilot'
    result = folder / (prefix + '.json.gz')
    source = DONOR if resume is None else resume
    origin = _read(source).get('tt19_online', {}).get('donor_sha256', _hash(source))
    report = dict(schema='tt19-so101-online-v1', seed=seed, fixture=spec, mode=mode, view=view,
                  donor_sha256=origin, starting_checkpoint_sha256=_hash(source), source_flight_sha256=_hash(SOURCE_FLIGHT),
                  track_coordinates='metres [side, towards rival, above tabletop]', games=[])
    tries = _seeds(seed, 17119, 0, games + game_offset)['tests'][game_offset:]
    seen = set()
    try:
        wiring = SO101Wiring(arena, spec['layout_seed'])
        state, player, transport = startup(source, arena.action_size, court, view, mode, seed, solver)
        report['solver'] = player.body.solver
        report['game_offset'] = game_offset
        report['setup'] = calibration(player, arena, wiring, seed, spec['setup_seed'], setup_frames)
        for number, game_seed in enumerate(tries, 1):
            score = ServiceScore(first_server=-1)
            rows = []
            for point, phase in schedule(game_seed):
                if mode == 'frozen' and (number > 1 or len(rows) == 2):
                    player.body.learning = player.explore = False
                if point in seen:
                    raise ValueError('reused a scored point seed')
                seen.add(point)
                sampled = player.body.samples
                raw = _point_row(player, arena, wiring, game_seed, point, phase, score.server)
                row = {key: raw[key] for key in ROW_FIELDS}
                row.update(optical_trace=player.trace, sightings=player.tracker.sightings,
                           optical_rate_frames=raw['exploration_rate_frames'], vision_digest=player.tracker.digest.hexdigest(),
                           samples_added=player.body.samples - sampled, observed_shots=raw['shots_observed'])
                if 'pitch_registration' in raw:
                    row['pitch_registration'] = raw['pitch_registration']
                rows.append(row)
                score.record(row['evaluator'])
                if limit is not None:
                    print(prefix, number, len(rows), score.score, row['server'], row['summary']['reason'],
                          row['evaluator']['service_contact'], row['player_returns'], row['samples_added'],
                          np.sum(player.tracker.sightings, axis=0).tolist(), flush=True)
                if score.finished or (limit is not None and len(rows) == limit):
                    break
            if not score.finished and limit is None:
                raise ValueError('the game exceeded the 65-point schedule')
            checkpoint = folder / f'{prefix}-game{number}-player.json.gz'
            save_checkpoint(checkpoint, state, player, transport, mode, origin)
            report['games'].append(dict(seed=game_seed, points=rows, score=list(score.score), finished=score.finished,
                                        metrics=metrics(rows), checkpoint=checkpoint.name,
                                        checkpoint_sha256=_hash(checkpoint)))
            _write(result, report)
            print(prefix, number, score.score, {key: report['games'][-1]['metrics'][key] for key in (
                'legal_own_serves', 'own_serves', 'total_legal_returns', 'two_return_points', 'frames')}, flush=True)
            if number < games:
                state, player, transport = startup(checkpoint, arena.action_size, court, view, mode, seed)
    finally:
        arena.close()
    report['replayed_points'] = verify(report)
    _write(result, report)
    return result


def metrics(points):
    game = _game_summary(points)
    clock = 0
    first = {'legal_serve': None, 'legal_return': None, 'second_legal_return': None}
    for row in points:
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
                first[kind] = clock + event['step']
        clock += row['steps']
    game.update(first_frames=first, frames=clock,
                paddle_releases=sum(event['kind'] == 'grip_release' for row in points for event in row['events']))
    return game


def verify(report):
    arena = SO101ViewArena(**report['fixture']['arena'])
    total = 0
    try:
        wiring = SO101Wiring(arena, report['fixture']['layout_seed'])
        setup = report['setup']
        if setup['frames']:
            interface = PointInterface(wiring, prepare_point(arena, 'ordinary'), server=-1, seed=report['seed'])
            interface.reset(seed=setup['seed'], render=False)
            for action in setup['actions']:
                interface.step(action, render=False)
            if interface.applied != setup['applied'] or len(setup['actions']) != setup['frames']:
                raise ValueError('the charged setup tape diverged')
        for game in report['games']:
            score = ServiceScore(first_server=-1)
            starts = islice(schedule(game['seed']), len(game['points']))
            for row, (point, phase) in zip(game['points'], starts, strict=True):
                if (row['seed'], row['server'], row['phase']) != (point, score.server, phase):
                    raise ValueError('scored order differs from replay')
                if not (len(row['actions']) == row['steps']
                        and len(row['optical_trace']) == len(row['sightings']) == row['steps'] + 1):
                    raise ValueError('an optical or motor frame is missing from the saved point')
                interface = PointInterface(wiring, prepare_point(arena, phase), server=score.server, seed=game['seed'])
                interface.reset(seed=point, render=False)
                for action in row['actions']:
                    interface.step(action, render=False)
                digest = sha256(np.asarray(row['actions'], dtype='<f8').tobytes()).hexdigest()
                if (arena.point_result() != row['evaluator'] or interface.applied != row['applied']
                        or digest != row['action_sha256']
                        or json.dumps(arena.events) != json.dumps(row['events'])):
                    raise ValueError('optical policy action tape differs in physical replay')
                score.record(arena.point_result())
                total += 1
            if list(score.score) != game['score'] or score.finished != game['finished']:
                raise ValueError('game score differs in replay')
    finally:
        arena.close()
    return total


def verify_rgb(receipt, *, limit=None, resume=None):
    report = _read(receipt)
    spec = report['fixture']
    mode, seed, view = (report[key] for key in ('mode', 'seed', 'view'))
    court = Court(**spec['court'])
    arena = SO101ViewArena(**spec['arena'])
    source = DONOR if resume is None else resume
    total = 0
    try:
        if _hash(source) != report['starting_checkpoint_sha256']:
            raise ValueError('starting player digest differs from the replay')
        wiring = SO101Wiring(arena, spec['layout_seed'])
        _, player, _ = startup(source, arena.action_size, court, view, mode, seed, report.get('solver', 'clipped'))
        setup = report['setup']
        if setup != calibration(player, arena, wiring, seed, spec['setup_seed'], setup.get('allotted', 0)):
            raise ValueError('camera motor preparation differs from saved setup')
        for index, game in enumerate(report['games']):
            for offset, row in enumerate(game['points']):
                if limit is not None and total == limit:
                    return total
                if mode == 'frozen' and (index or offset == 2):
                    player.body.learning = player.explore = False
                raw = _point_row(player, arena, wiring, game['seed'], row['seed'], row['phase'], row['server'])
                if (raw['action_sha256'] != row['action_sha256']
                        or player.tracker.digest.hexdigest() != row['vision_digest']
                        or player.trace != row['optical_trace']
                        or raw.get('pitch_registration') != row.get('pitch_registration')):
                    raise ValueError(f"optical policy replay differs on point {row['seed']}")
                total += 1
            checkpoint = receipt.parent / game['checkpoint']
            if _hash(checkpoint) != game['checkpoint_sha256']:
                raise ValueError('game checkpoint digest differs')
            state = _read(checkpoint)['tt19_online']
            stored = state['motor']
            restored = player.body.state()
            if 'solver' not in stored:
                restored.pop('solver')
            if restored != stored:
                raise ValueError('optical updates did not reproduce the saved motor state')
            if 'pitch_registration' in state and (player.registration != state['pitch_registration']
                                                   or player.registration_count != state['pitch_updates']):
                raise ValueError('optical pitch feedback did not reproduce the saved stroke state')
            _, player, _ = startup(checkpoint, arena.action_size, court, view, mode, seed)
    finally:
        arena.close()
    return total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--mode', choices=MODES, default='warm')
    parser.add_argument('--view', default='wrist+fixed')
    parser.add_argument('--games', type=int, default=2)
    parser.add_argument('--setup-frames', type=int, default=0)
    parser.add_argument('--solver', choices=('bounded', 'clipped', 'drive', 'register'), default='bounded')
    parser.add_argument('--game-offset', type=int, default=0)
    parser.add_argument('--strict-endline', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--rgb', action='store_true')
    args = parser.parse_args()
    if args.verify:
        replayed = (verify_rgb(args.verify, limit=args.limit, resume=args.resume)
                    if args.rgb else verify(_read(args.verify)))
        print('replayed', replayed)
    else:
        play(args.folder, seed=args.seed, mode=args.mode, view=args.view, games=args.games,
             setup_frames=args.setup_frames, limit=args.limit, resume=args.resume, solver=args.solver,
             game_offset=args.game_offset, strict=args.strict_endline)
