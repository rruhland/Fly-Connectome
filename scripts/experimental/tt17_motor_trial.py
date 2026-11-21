"""Play full TT17 table games with frozen signed, TT15, or bounded motor maps."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_continuous_trial import prepare_point
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import ServiceScore
from .tt8_service_trial import PointInterface
from .tt10_series_trial import _game_summary
from .tt10_window_trial import _point_row
from .tt14_gain_model import GainMemory
from .tt14_gain_servo import GainServo
from .tt14_gain_trial import save_player
from .tt15_motor_probe import ProbeMemory, measure, pulse_commands
from .tt16_transfer_bridge import costs
from .tt16_transfer_game import ROW_FIELDS
from .tt17_motor_fit import Refinement, initialise
from .tt17_motor_protocol import CASES, load_optical_player, make_arena, setup_path, wiring_for


MODES = ('signed', 'tt15', 'bounded')
ALL_MODES = (*MODES, 'bounded_frozen')


def prefix(folder, case, label):
    return folder / f'motor-{case}-{label}'


def startup(spec, archived, mode, initial_motor=None):
    if mode == 'tt15':
        recalled = GainMemory(archived.source, 'hybrid', archived.models[1:])
        return ProbeMemory(recalled, spec['tt15_reference']), None, dict(reason='TT15 in-game confirmation')
    if initial_motor is None:
        signed, bounded, signal, evidence = initialise(archived.source, spec['actions'], spec['reference'], spec['active'])
        chosen = signed if mode == 'signed' else bounded
    else:
        if mode != 'bounded_frozen':
            raise ValueError('supply an explicit motor only for the frozen mode')
        chosen, signal, evidence = initial_motor, None, dict(reason='externally fitted optics')
    memory = GainMemory(archived.source, 'hybrid')
    if chosen.base is not archived.source or chosen.coupled:
        memory.maps.append(chosen)
        memory.active = 1
    return memory, Refinement(chosen, signal, mode == 'bounded'), evidence


def update(mode, memory, refine, actions, trace, acquire):
    if mode == 'tt15':
        return memory.observe(actions, trace, acquire)
    decision = refine.observe(actions, trace)
    if decision['adopted']:
        memory.maps.append(refine.motor)
        memory.active = len(memory.maps) - 1
    return decision


def describe_map(motor):
    width = motor.base.action_size
    return dict(kind=motor.kind, transform=motor.transform.tolist(),
                camera_jacobian=motor.base.jacobian(np.zeros(width)).tolist())


def tally(rows, setup_frames, clock_dt):
    before, game_frames, dry_frames = setup_frames, 0, 0
    first = None
    for row in rows:
        if first is None and row['first_legal_return_step'] is not None:
            first = before + row['first_legal_return_step']
        later = 0 if row['probe'] is None else len(row['probe']['poses']) - 1
        game_frames += row['steps']
        dry_frames += later
        before += row['steps'] + later
    return dict(setup=setup_frames, game=game_frames, additional_dry=dry_frames,
                total=before, first_legal_return=first, first_return_seconds=None if first is None else first * clock_dt,
                points_per_1000_frames=1000 * sum(row['points_for'] for row in rows) / before)


def replay_updates(report, spec, arena, initial_motor):
    _, _, player, archived = load_optical_player(spec, arena)
    mode = report['mode']
    memory, refinement, evidence = startup(spec, archived, mode, initial_motor)
    if report['initial_map'] != describe_map(memory.maps[memory.active]) or report['optical_fit'] != evidence:
        raise ValueError('the initial optical decision changed')
    wiring = wiring_for(arena, spec)
    count = 0
    for point in report['games'][0]['points']:
        dry = point['probe']

        def acquire():
            if dry is None:
                raise ValueError('decision replay requested a missing dry movement')
            poses = measure(wiring, dry['seed'], player.tracker_factory)
            if poses != dry['poses']:
                raise ValueError('repeated TT15 camera history changed')
            return poses

        if point['model_used'] != memory.active:
            raise ValueError('wrong model used for scored start')
        camera = [dict(blade=p, blade_pitch=a) for p, a in point.get('motor_sightings', [])]
        outcome = update(mode, memory, refinement, point['actions'], camera, acquire)
        if json.loads(json.dumps(outcome)) != json.loads(json.dumps(point['choice'])):
            raise ValueError('optical update differs from archived decision')
        count += dry is not None
    if report['final_map'] != describe_map(memory.maps[memory.active]):
        raise ValueError('camera-only updates ended at a different model')
    return int(count)


def verify_physics(report, spec):
    arena = make_arena(spec)
    try:
        wires = wiring_for(arena, spec)
        score = ServiceScore(first_server=-1)
        game = report['games'][0]
        for row, point, phase in zip(game['points'], spec['starts'], spec['phases']):
            if score.finished or (row['server'], row['seed'], row['phase']) != (score.server, point, phase):
                raise ValueError('score or point schedule differs from the saved game')
            interface = PointInterface(wires, prepare_point(arena, phase), server=score.server, seed=game['seed'])
            interface.reset(seed=point, render=False)
            for command in row['actions']:
                interface.step(command, render=False)
            if (arena.point_result() != row['evaluator'] or interface.applied != row['applied']
                    or json.loads(json.dumps(arena.events)) != row['events']
                    or sha256(np.asarray(row['actions'], dtype='<f8').tobytes()).hexdigest() != row['action_sha256']):
                raise ValueError('scored motion did not replay')
            score.record(arena.point_result())
        if not score.finished or list(score.score) != game['score']:
            raise ValueError('saved score is not a complete replayed game')
    finally:
        arena.close()


def export_initial(folder, case):
    setup = setup_path(folder, case)
    spec = _read(setup)
    arena = make_arena(spec)
    path = prefix(folder, case, 'bounded-start-player').with_suffix('.json.gz')
    try:
        state, library, player, archived = load_optical_player(spec, arena)
        memory, _, optics = startup(spec, archived, 'bounded_frozen')
        save_player(path, state, library, player, memory)
        saved = _read(path)
        saved['tt17_motor'] = dict(mode='bounded_start', setup_sha256=_hash(setup),
                                   optical_fit=optics, calibration_frames=spec['optics_frames'] + len(spec['actions']))
        _write(path, saved)
    finally:
        arena.close()
    return path


def play(folder, case, mode, *, configure=None, label=None, initial_motor=None, preparation_override=None,
         servo_factory=None):
    if mode not in ALL_MODES:
        raise ValueError('unknown calibration method')
    setup = setup_path(folder, case)
    spec = _read(setup)
    arena = make_arena(spec)
    tag = mode if label is None else label
    target = prefix(folder, case, tag)
    path = target.with_suffix('.json.gz')
    checkpoint = target.with_name(f'{target.name}-player.json.gz')
    if preparation_override is not None:
        preparation = preparation_override
    else:
        preparation = dict(optics=costs([a for p in spec['optics_probes'] for a in p['actions']]))
        if mode == 'tt15':
            frames = len(spec['tt15_reference']) - 1
            preparation['reference_dry'] = costs(pulse_commands(len(spec['matrix']))[:frames])
        else:
            preparation['active'] = costs(spec['actions'])
    setup_frames = sum(item['frames'] for item in preparation.values())
    try:
        cache_hall_stereo(arena, reflections=True)
        state, library, player, archived = load_optical_player(spec, arena)
        memory, refine, optics = startup(spec, archived, mode, initial_motor)
        initial = describe_map(memory.maps[memory.active])
        if configure is not None:
            configure(player)
        inherited_servo = player._make_servo
        physical_servo = GainServo if servo_factory is None else servo_factory
        score = ServiceScore(first_server=-1)
        wires = wiring_for(arena, spec)
        points = []
        for point, phase, dry_seed in zip(spec['starts'], spec['phases'], spec['followups'], strict=True):
            motor = memory.maps[memory.active]
            player.adapter = motor.base
            player._make_servo = (lambda: physical_servo(motor, player.timing_memory, learn=player.adjust)
                                  ) if motor.coupled else inherited_servo
            row = _point_row(player, arena, wires, spec['game_seed'], point, phase, score.server)
            compact = {field: row[field] for field in ROW_FIELDS}
            dry = None

            def acquire():
                nonlocal dry
                dry = dict(seed=dry_seed, poses=measure(wires, dry_seed, player.tracker_factory))
                return dry['poses']

            before = memory.active
            chosen = update(mode, memory, refine, row['actions'], player.trace, acquire)
            compact.update(changed=True, model_used=before, choice=chosen, probe=dry)
            if mode in ('tt15', 'bounded'):
                compact['motor_sightings'] = [[frame['blade'], frame['blade_pitch']] for frame in player.trace]
            points.append(compact)
            score.record(row['evaluator'])
            if score.finished:
                break
        if not score.finished:
            raise ValueError('table game did not finish')
        save_player(checkpoint, state, library, player, memory)
        saved = _read(checkpoint)
        saved['tt17_motor'] = dict(mode=mode, setup_sha256=_hash(setup), initial=initial, optical_fit=optics)
        _write(checkpoint, saved)
        game = dict(seed=spec['game_seed'], points=points, score=list(score.score), metrics=_game_summary(points))
        report = dict(schema='tt17-motor-v1', case=case, mode=mode, label=tag, arena=spec['arena'], camera=spec['camera'],
                      start=spec['donor_sha256'], setup_sha256=_hash(setup), matrix=spec['matrix'], games=[game],
                      preparation=preparation, optical_fit=optics, initial_map=initial,
                      final_map=describe_map(memory.maps[memory.active]), checkpoint=_hash(checkpoint),
                      clock=tally(points, setup_frames, spec['control_dt']))
        _write(path, report)
        report = _read(path)
        report['replayed_dry'] = replay_updates(report, spec, arena, initial_motor)
    finally:
        arena.close()
    _write(path, report)
    verify_physics(_read(path), spec)
    report['replayed_points'] = len(points)
    _write(path, report)
    print(case, tag, score.score, game['metrics']['total_legal_returns'], game['metrics']['two_return_points'],
          optics['reason'], report['clock']['additional_dry'], flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('case', choices=CASES)
    parser.add_argument('mode', choices=(*ALL_MODES, 'initial'))
    args = parser.parse_args()
    if args.mode == 'initial':
        print(export_initial(args.folder, args.case))
    else:
        play(args.folder, args.case, args.mode)
