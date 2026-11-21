"""Play complete alternating-service games through hidden motor changes and restoration."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import process_time_ns

import numpy as np

from .tt4_transfer_trial import _hash, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import match_phases, prepare_point
from .tt6_memory import load
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_trial import PointInterface
from .tt10_series_trial import PHASES, _game_summary, _restore, _save
from .tt10_window_trial import _point_row
from .tt14_gain_hardware import GainWiring, perturbation
from .tt14_gain_model import GainMemory, MotorMap, Observation
from .tt14_gain_servo import GainServo


def load_map(path):
    """Return the active camera body and the learned input-to-body transform."""
    state = _read(path)
    library = load(path)[0]
    memory = state.get('gain_memory')
    if memory is None:
        model = library.models[state['series']['model']]
        return model, np.eye(model.action_size)
    active = memory['maps'][memory['active']]
    return library.models[active['model']], np.asarray(active['transform'])


def memory_from_checkpoint(state, library, mode):
    source = library.models[state['series']['model']]
    inherited = state.get('port_memory')
    prior = [library.models[index] for index in inherited['indices'] if library.models[index] is not source
             ] if inherited else []
    memory = GainMemory(source, mode, prior)
    saved = state.get('gain_memory')
    if saved:
        memory.maps = [MotorMap(library.models[row['model']], np.asarray(row['transform']), row['kind'])
                       for row in saved['maps']]
        memory.source = memory.maps[0].base
        memory.active = saved['active']
        memory.previous = Observation.restore(saved['previous']) if saved['previous'] is not None else None
        memory.support = [Observation.restore(row) for row in saved['support']]
    return memory


def save_player(path, state, library, player, memory):
    maps = []
    for motor in memory.maps:
        index = next((i for i, model in enumerate(library.models) if model is motor.base), None)
        if index is None:
            index = len(library.models)
            library.models.append(motor.base)
        maps.append(dict(model=index, transform=motor.transform.tolist(), kind=motor.kind))
    entry = {k: v for k, v in state['series'].items() if k not in
             ('completed', 'role_rows', 'flight_history', 'flight_impulses')}
    entry['model'] = maps[memory.active]['model']
    _save(path, state, library, player, entry, state['series']['completed'] + 1)
    portable = _read(path)
    portable['gain_memory'] = dict(active=memory.active, maps=maps,
                                  previous=memory.previous.state() if memory.previous is not None else None,
                                  support=[obs.state() for obs in memory.support])
    _write(path, portable)


def wiring_for(arena, receipt, changed):
    seed = receipt['arena']['seed']
    return GainWiring(arena, seed, receipt['matrix']) if changed else TT6Wiring(arena, seed)


def verify(receipt, source=None):
    """Replay every point on the physical apparatus, then replay the recorded updates."""
    arena = TT8ServiceArena(**receipt['arena'])
    try:
        wires = {changed: wiring_for(arena, receipt, changed) for changed in (False, True)}
        total = 0
        memory = None
        if source is not None:
            if _hash(source) != receipt['start']:
                raise ValueError('initial player differs from the game receipt')
            memory = memory_from_checkpoint(_read(source), load(source)[0], receipt['mode'])
        for game in receipt['games']:
            score = ServiceScore(first_server=-1)
            for row in game['points']:
                if score.finished or score.server != row['server']:
                    raise ValueError('alternating service order diverged')
                interface = PointInterface(wires[row['changed']], prepare_point(arena, row['phase']),
                                           server=score.server, seed=game['seed'])
                interface.reset(seed=row['seed'], render=False)
                for command in row['actions']:
                    interface.step(command, render=False)
                if arena.point_result() != row['evaluator'] or interface.applied != row['applied']:
                    raise ValueError(f"physics replay diverged on point {row['seed']}")
                if json.loads(json.dumps(arena.events)) != row['events']:
                    raise ValueError(f"physics event replay diverged on point {row['seed']}")
                digest = sha256(np.asarray(row['actions'], dtype='<f8').tobytes()).hexdigest()
                if digest != row['action_sha256']:
                    raise ValueError(f"action digest diverged on point {row['seed']}")
                if memory is not None:
                    trace = [dict(blade=xyz, blade_pitch=pitch) for xyz, pitch in row['motor_sightings']]
                    choice = memory.observe(row['actions'], trace)
                    if json.loads(json.dumps(choice)) != row['choice']:
                        raise ValueError(f"camera-only update diverged on point {row['seed']}")
                score.record(arena.point_result())
                total += 1
            if not score.finished or list(score.score) != game['score'] or _game_summary(game['points']) != game['metrics']:
                raise ValueError('recorded game score differs from replayed points')
        return total
    finally:
        arena.close()


def run(start, prefix, mode, seed, perturbation_seed, profile, games, switch):
    arena_info = _read(start)['series']['arena']
    arena = TT8ServiceArena(**arena_info)
    output = prefix.with_suffix('.json.gz')
    output.parent.mkdir(exist_ok=True, parents=True)
    observed_matrix = perturbation(arena.action_size, perturbation_seed, profile)
    data = dict(schema='tt14-gain-v1', arena=arena_info, start=_hash(start), mode=mode, seed=seed, profile=profile,
                perturbation_seed=perturbation_seed, matrix=observed_matrix.tolist(), switch=switch,
                implementation={name: _hash(Path(__file__).with_name(f'tt14_gain_{name}.py'))
                                for name in ('trial', 'hardware', 'model', 'servo')}, games=[])
    current = start
    try:
        cache_hall_stereo(arena, reflections=True)
        wires = {changed: wiring_for(arena, data, changed) for changed in (False, True)}
        for game_index, game_seed in enumerate(_seeds(seed, 9387, 0, games)['tests']):
            state, library, player = _restore(current, arena)
            memory = memory_from_checkpoint(state, library, mode)
            default_servo = player._make_servo
            starts = _seeds(game_seed, 6011, 0, 65)['tests']
            phases = match_phases(PHASES, 65, [game_seed, 6011])
            score = ServiceScore(first_server=-1)
            rows = []
            for i, (point_seed, phase) in enumerate(zip(starts, phases, strict=True)):
                changed = profile != 'stable' and ((game_index == 0) == (i >= switch))
                motor = memory.maps[memory.active]
                player.adapter = motor.base
                player._make_servo = (lambda: GainServo(motor, player.timing_memory, learn=player.adjust)
                                      ) if motor.coupled else default_servo
                row = _point_row(player, arena, wires[changed], game_seed, point_seed, phase, score.server)
                before = process_time_ns()
                choice = memory.observe(row['actions'], player.trace)
                elapsed = process_time_ns() - before
                row.update(changed=changed, model_used=choice['previous'], model_kind=motor.kind,
                           choice=choice, update_cpu_ns=elapsed,
                           motor_sightings=[(frame['blade'], frame['blade_pitch']) for frame in player.trace])
                rows.append(row)
                score.record(row['evaluator'])
                if choice['active'] != choice['previous']:
                    print(f'{mode} game {game_index + 1} point {i + 1}: {choice}', flush=True)
                if score.finished:
                    break
            if not score.finished:
                raise ValueError('game exceeded 65 points')
            stage = ('stable' if profile == 'stable' else 'changed') if game_index == 0 else 'restored'
            current = prefix.with_name(f'{prefix.name}-{stage}-player.json.gz')
            save_player(current, state, library, player, memory)
            data['games'].append(dict(seed=game_seed, score=score.score, points=rows, metrics=_game_summary(rows),
                                      checkpoint=_hash(current), active=memory.active,
                                      active_kind=memory.maps[memory.active].kind))
            _write(output, data)
            print(mode, stage, score.score, len(rows), data['games'][-1]['metrics'], flush=True)
    finally:
        arena.close()
    data['replayed_points'] = verify(_read(output), start)
    _write(output, data)
    print('physical and camera-only replays', data['replayed_points'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('start', type=Path)
    parser.add_argument('prefix', type=Path)
    parser.add_argument('--mode', choices=('fixed', 'remap', 'refit', 'hybrid'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--perturbation-seed', type=int)
    parser.add_argument('--profile', choices=('gain', 'coupling', 'stable'))
    parser.add_argument('--games', type=int, choices=(1, 2), default=2)
    parser.add_argument('--switch', type=int, default=4)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(verify(_read(args.prefix.with_suffix('.json.gz')), args.start))
    elif any(value is None for value in (args.mode, args.seed, args.perturbation_seed, args.profile)):
        parser.error('specify the mode, seed, perturbation seed and profile')
    else:
        run(args.start, args.prefix, args.mode, args.seed, args.perturbation_seed, args.profile, args.games, args.switch)
