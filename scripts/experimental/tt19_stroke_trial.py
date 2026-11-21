"""Learn direct SO-101 strokes from native RGB and point rewards, then freeze them."""

import argparse
from dataclasses import asdict
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
from .tt8_service_arena import ServiceScore
from .tt8_service_trial import PointInterface
from .tt19_stroke import RallyStrokePoint, StrokePoint, StrokeSearch, gantry_style, lateral_gain, track_coordinates
from .tt19_view import MODES, ViewTransport


DONOR = ROOT / 'artifacts' / 'tt17' / 'motor-gantry-coupling-travel-player.json.gz'
TAPES = tuple(ROOT / 'artifacts' / 'tt18' / f'combined-{seed}-carried.json.gz'
              for seed in (418336241, 349051097))


class IdentityWiring:
    def __init__(self, arena):
        self.arena = arena

    def reset(self, **kwargs):
        return self.arena.reset(**kwargs)

    def step(self, action, *, render=True):
        return self.arena.step(action, render=render)


def _arena(settings, cameras, table):
    from .tt19_view_trial import SO101ViewArena

    if table is not None:
        from .tt18_plant import PlantSettings

        return SO101ViewArena(view_mode=cameras, plant=PlantSettings(table_half_width=table[0],
                                                                    table_half_length=table[1]), **settings)
    return SO101ViewArena(view_mode=cameras, **settings)


def motion_probes(arena, transport, seed):
    """Build a discrete catalogue of forward and diagonal camera-observed pulses."""
    commands = []
    for axes in (*[(axis,) for axis in (1, 2, 3)], *combinations((1, 2, 3), 2)):
        for signs in product((-1, 1), repeat=len(axes)):
            command = np.zeros(6)
            command[list(axes)] = np.array(signs) * (.77 if len(axes) == 1 else .57)
            commands.append(command)
    proposals = []
    for command in commands:
        tracker = transport.tracker()
        tracker.update(arena.reset(seed=seed, server=1))
        before = track_coordinates(tracker.actor)
        for tick in range(5):
            frames, _, ended, truncated, _ = arena.step(command, render=tick in (2, 4))
            if frames is not None:
                tracker.update(frames)
            if ended or truncated:
                break
        after = track_coordinates(tracker.actor)
        delta = None if before is None or after is None else (np.subtract(after, before) / ((tick + 1) * arena.control_dt)).tolist()
        proposals.append(dict(action=command.tolist(), optical_start=before, optical_end=after,
                              commands=tick + 1, velocity=delta))
    return proposals


def play(arena, transport, search, seed, game_seed, server, *, train, forecast=None, stride=1, rearm=False,
         pan_gain=None):
    interface = PointInterface(IdentityWiring(arena), server=server, seed=game_seed)
    stroke = search.choose(server, train=train)
    tracker = transport.tracker()
    rally = rearm or pan_gain is not None
    if rally:
        receiver = stroke if server == -1 else search.choose(-1, train=False)
        player = RallyStrokePoint(stroke, receiver, tracker, server, arena.control_dt,
                                  forecast=forecast, rearm=rearm, pan_gain=pan_gain)
    else:
        player = StrokePoint(stroke, tracker, server, arena.control_dt, forecast=forecast)
    frame = interface.reset(seed=seed)
    player.observe(frame)
    sources = [tracker.sources.copy()]
    commands = []
    while True:
        action = player.command()
        commands.append(list(action))
        frame, reward, terminated, truncated, _ = interface.step(action, render=len(commands) % stride == 0)
        player.observe(frame)
        sources.append(None if frame is None else tracker.sources.copy())
        if terminated or truncated:
            break
    swings = player.swing_ticks if rally else [] if player.swing_tick is None else [player.swing_tick]
    detail = search.finish(stroke, server, reward, player.observations, arena.control_dt, next(iter(swings), None),
                           train=train)
    return dict(seed=seed, game_seed=game_seed, server=server, stroke=asdict(stroke), ticks=len(commands),
                actions=commands, applied=interface.applied, public_reward=reward, optical=player.observations,
                swing_tick=next(iter(swings), None), swing_ticks=swings, rendered_frames=len(tracker.sightings),
                observation_sha256=tracker.digest.hexdigest(), optical_sources=sources,
                sightings=np.sum(tracker.sightings, axis=0).tolist(), detail=detail,
                evaluator=arena.point_result(), events=arena.events.copy())


def summary(rows):
    result = {}
    for server, role in ((1, 'serve'), (-1, 'return')):
        subset = [row for row in rows if row['server'] == server]
        frames = sum(row['ticks'] + 1 for row in subset)
        result[role] = dict(attempts=len(subset), frames=frames,
                            captured_frames=sum(row['rendered_frames'] for row in subset),
                            ball_seen_frames=sum(row['sightings'][1] for row in subset),
                            blade_seen_frames=sum(row['sightings'][2] for row in subset),
                            swings=sum(row['swing_tick'] is not None for row in subset),
                            second_swings=sum(len(row.get('swing_ticks') or ()) >= 2 for row in subset),
                            optical_contacts=sum(bool(row['detail']['contact']) for row in subset),
                            legal=sum(row['evaluator']['legal_serve'] if server == 1
                                      else row['evaluator']['legal_returns'] for row in subset),
                            scored=sum(row['public_reward'] > 0 for row in subset),
                            legal_returns=sum(row['evaluator']['legal_returns'] for row in subset),
                            sustained=sum(row['evaluator']['legal_returns'] >= 2 for row in subset),
                            timeouts=sum(row['evaluator']['reason'] == 'timeout' for row in subset))
    return result


def verify_physics(report):
    arena = _arena(report['arena'], report['cameras'], report['custom_table'])
    count = 0
    try:
        for rows in (report['practice'], report['test']):
            for row in rows:
                interface = PointInterface(IdentityWiring(arena), server=row['server'], seed=row['game_seed'])
                interface.reset(seed=row['seed'], render=False)
                for action in row['actions']:
                    interface.step(action, render=False)
                if (interface.applied != row['applied'] or arena.point_result() != row['evaluator']
                        or json.dumps(arena.events, sort_keys=True) != json.dumps(row['events'], sort_keys=True)):
                    raise ValueError(f"physical replay diverged for point {row['seed']}")
                count += 1
        for game in report.get('games', []):
            board = ServiceScore(first_server=-1)
            rows = [row for row in report['test'] if row['test_game'] == game['index']]
            for row in rows:
                board.record(row['evaluator'])
            if list(board.score) != game['score'] or board.finished != game['finished'] or summary(rows) != game['metrics']:
                raise ValueError('replayed point outcomes differ from the recorded game')
    finally:
        arena.close()
    return count


def run(args):
    styles, samples = {}, []
    lineage = None
    if args.mode == 'gantry':
        lineage = DONOR if args.resume is None else args.resume
        rows = [{key: row[key] for key in ('server', 'public_reward', 'optical')}
                for path in TAPES for row in _read(path)['points']]
        styles, samples = gantry_style(rows, 1. / 30.)
    saved = {} if lineage is None else _read(lineage)
    forecast = None
    if args.flight:
        from .tt19_flight import FlightLaw

        forecast = FlightLaw.from_state(_read(args.flight)['source_law'])
    calibrated_pan = None
    if args.pan_from:
        gain, samples_used = lateral_gain(_read(args.pan_from)['practice'])
        calibrated_pan = dict(gain_m_per_command=gain, samples=samples_used, practice_sha256=_hash(args.pan_from))
    settings = dict(seed=args.seed, width=args.width, height=args.height, max_steps=args.max_steps,
                    frameskip=8, venue='hall', detail=args.detail)
    if args.strict_endline:
        settings['strict_endline'] = True
    arena = _arena(settings, args.cameras, args.table)
    transport = ViewTransport(args.cameras)
    search = StrokeSearch(args.seed, source_style=styles)
    receipt = dict(schema='tt19-direct-stroke-v1', mode=args.mode, seed=args.seed, arena=settings,
                   cameras=args.cameras, custom_table=args.table, control_dt=arena.control_dt,
                   capture_stride=args.stride, donor_sha256=None if lineage is None else _hash(lineage),
                   optical_columns=['ball_xyz_m', 'blade_xyz_m', 'blade_pitch_rad'],
                   optical_origin='tabletop center', source_tapes={path.name: _hash(path) for path in TAPES}
                   if args.mode == 'gantry' else {},
                   flight_source_sha256=None if not args.flight else _hash(args.flight),
                   source_style=styles, source_optical_strokes=samples, practice=[], test=[], rearm=args.rearm,
                   pan_calibration=calibrated_pan)
    if args.resume:
        state = _read(args.resume)['tt19_stroke']
        if state['mode'] != args.mode:
            raise ValueError('resume mode differs from incoming stroke policy')
        search = StrokeSearch.restore(state['policy'])
        transport = ViewTransport(args.cameras, state=state['rgb_calibration'])
    if args.placement_source:
        placement = [{key: row[key] for key in ('server', 'stroke', 'detail', 'public_reward')}
                     for row in _read(args.placement_source)['practice']]
        search.prefer_placement(placement)
        receipt['placement_source_sha256'] = _hash(args.placement_source)
    schedule = _seeds(args.seed, 6874, args.practice, args.test)
    pan_gain = None if calibrated_pan is None else calibrated_pan['gain_m_per_command']
    try:
        new_probes = []
        if args.probes and not search.probes:
            new_probes = search.probes = motion_probes(arena, transport, args.seed + 100)
        receipt['probe_commands'] = sum(row['commands'] for row in new_probes)
        receipt['probes'] = new_probes
        print('movement probes', receipt['probe_commands'], 'valid',
              sum(row['velocity'] is not None for row in search.probes), flush=True)
        _write(args.output, receipt)
        for index, point_seed in enumerate(schedule['calibration']):
            server = -1 if args.receive_only else 1 if args.serve_only or index % 2 == 0 else -1
            row = play(arena, transport, search, point_seed, args.seed, server, train=True, forecast=forecast,
                       stride=args.stride, rearm=args.rearm, pan_gain=pan_gain)
            receipt['practice'].append(row)
            if index == 0 or (index + 1) % 4 == 0 or row['public_reward'] > 0:
                print('practice', index + 1, server, row['swing_tick'], row['detail']['nearest_m'],
                      row['evaluator']['reason'], row['sightings'], flush=True)
            _write(args.output, receipt)
        receipt['games'] = []
        for game in range(args.games):
            board = ServiceScore(first_server=-1) if args.game else None
            seeds = schedule if game == 0 else _seeds(args.seed, 6874 + game, 0, args.test)
            current = []
            for index, point_seed in enumerate(seeds['tests']):
                server = (board.server if board is not None else -1 if args.receive_only
                          else 1 if args.serve_only or index % 2 == 0 else -1)
                row = play(arena, transport, search, point_seed, args.seed + 1 + game, server, train=False,
                           forecast=forecast, stride=args.stride, rearm=args.rearm, pan_gain=pan_gain)
                row['test_game'] = game
                receipt['test'].append(row)
                current.append(row)
                if board is not None:
                    board.record(row['evaluator'])
                    receipt['game_score'] = list(board.score)
                    receipt['game_completed'] = board.finished
                print('test', game + 1, index + 1, server, row['swing_ticks'], row['detail']['nearest_m'],
                      row['evaluator']['reason'], row['sightings'], flush=True)
                _write(args.output, receipt)
                if board is not None and board.finished:
                    break
            if board is not None:
                receipt['games'].append(dict(index=game, score=list(board.score), finished=board.finished,
                                             metrics=summary(current)))
    finally:
        arena.close()
    checkpoint = args.output.with_name(args.output.name.replace('.json.gz', '-player.json.gz'))
    saved['tt19_stroke'] = dict(policy=search.state(), rgb_calibration=transport.state(), mode=args.mode, rearm=args.rearm,
                                gantry_checkpoint_sha256=receipt['donor_sha256'], pan_calibration=calibrated_pan)
    _write(checkpoint, saved)
    restored = _read(checkpoint)['tt19_stroke']['policy']
    assert json.loads(json.dumps(StrokeSearch.restore(restored).state())) == restored
    receipt.update(checkpoint=checkpoint.name, checkpoint_sha256=_hash(checkpoint),
                   practice_summary=summary(receipt['practice']), test_summary=summary(receipt['test']))
    receipt['replayed_points'] = verify_physics(receipt)
    _write(args.output, receipt)
    print(receipt['practice_summary'], receipt['test_summary'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('mode', choices=('gantry', 'cold'))
    parser.add_argument('--seed', type=int, default=54916031)
    parser.add_argument('--practice', type=int, default=48)
    parser.add_argument('--test', type=int, default=28)
    roles = parser.add_mutually_exclusive_group()
    roles.add_argument('--serve-only', action='store_true')
    roles.add_argument('--receive-only', action='store_true')
    parser.add_argument('--game', action='store_true')
    parser.add_argument('--games', type=int, default=1)
    parser.add_argument('--rearm', action='store_true')
    parser.add_argument('--strict-endline', action='store_true')
    parser.add_argument('--width', type=int, default=288)
    parser.add_argument('--height', type=int, default=216)
    parser.add_argument('--max-steps', type=int, default=87)
    parser.add_argument('--stride', type=int, choices=(1, 2), default=2)
    parser.add_argument('--probes', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--detail', choices=('preview', 'cad'), default='preview')
    parser.add_argument('--table', type=float, nargs=2, metavar=('HALF_WIDTH', 'HALF_LENGTH'))
    parser.add_argument('--cameras', choices=MODES, default='fixed')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--placement-source', type=Path)
    parser.add_argument('--flight', type=Path)
    parser.add_argument('--pan-from', type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.name.endswith('.json.gz'):
        parser.error('choose a .json.gz output receipt path')
    if args.games < 1 or args.games > 1 and not args.game:
        parser.error('multiple scheduled tests require scored game mode')
    run(args)


if __name__ == '__main__':
    main()
