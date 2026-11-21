"""Play through motor changes with optical confirmation on independent points."""

import argparse
import json
from pathlib import Path
from time import process_time_ns

from .tt4_transfer_trial import _hash, _read, _seeds, _write
from .tt6_continuous_trial import match_phases
from .tt6_memory import load
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt10_series_trial import PHASES, _game_summary, _restore
from .tt10_window_trial import _point_row
from .tt14_gain_hardware import perturbation
from .tt14_gain_servo import GainServo
from .tt14_gain_trial import memory_from_checkpoint, save_player, verify, wiring_for
from .tt15_motor_guard import GuardedMemory


def restore_memory(state, library):
    recalled = memory_from_checkpoint(state, library, 'hybrid')
    memory = GuardedMemory(recalled)
    if state.get('motor_guard'):
        memory.support = recalled.support
    return memory


def archive_point(memory, actions, sightings):
    trace = [dict(blade=blade, blade_pitch=pitch) for blade, pitch in sightings]
    return memory.observe(actions, trace)


def inspect(receipt, source):
    if _hash(source) != receipt['start']:
        raise ValueError('point audit requires the exact initial checkpoint')
    replayed = verify(receipt)
    memory = restore_memory(_read(source), load(source)[0])
    for game in receipt['games']:
        for row in game['points']:
            selected = archive_point(memory, row['actions'], row['motor_sightings'])
            if json.loads(json.dumps(selected)) != row['choice']:
                raise ValueError(f"camera-only update replay diverged on point {row['seed']}")
    return replayed


def run(start, prefix, seed, perturbation_seed, profile, games, switch):
    arena_settings = _read(start)['series']['arena']
    arena = TT8ServiceArena(**arena_settings)
    changed_matrix = perturbation(arena.action_size, perturbation_seed, profile)
    result = dict(schema='tt15-guarded-motor-v1', arena=arena_settings, start=_hash(start), mode='guard',
                  seed=seed, profile=profile, perturbation_seed=perturbation_seed, switch=switch,
                  matrix=changed_matrix.tolist(), games=[], implementation={
                      key: _hash(Path(__file__).with_name(f'tt15_motor_{key}.py')) for key in ('guard', 'trial')})
    output = prefix.with_suffix('.json.gz')
    output.parent.mkdir(exist_ok=True, parents=True)
    resume = start
    try:
        cache_hall_stereo(arena, reflections=True)
        wiring = {key: wiring_for(arena, result, key) for key in (False, True)}
        for game_index, game_seed in enumerate(_seeds(seed, 9387, 0, games)['tests']):
            state, library, player = _restore(resume, arena)
            memory = restore_memory(state, library)
            make_servo = player._make_servo
            score = ServiceScore(first_server=-1)
            seeds = _seeds(game_seed, 6011, 0, 65)['tests']
            phases = match_phases(PHASES, 65, [game_seed, 6011])
            points = []
            for point_index, (point_seed, phase) in enumerate(zip(seeds, phases, strict=True)):
                changed = profile != 'stable' and ((game_index == 0) == (point_index >= switch))
                motor = memory.maps[memory.active]
                player.adapter = motor.base
                player._make_servo = (lambda: GainServo(motor, player.timing_memory, learn=player.adjust)
                                      ) if motor.coupled else make_servo
                row = _point_row(player, arena, wiring[changed], game_seed, point_seed, phase, score.server)
                before = process_time_ns()
                decision = memory.observe(row['actions'], player.trace)
                elapsed = process_time_ns() - before
                row.update(changed=changed, model_used=decision['previous'], model_kind=motor.kind,
                           choice=decision, update_cpu_ns=elapsed,
                           motor_sightings=[(f['blade'], f['blade_pitch']) for f in player.trace])
                points.append(row)
                score.record(row['evaluator'])
                if decision['active'] != decision['previous']:
                    print(game_index + 1, point_index + 1, decision, flush=True)
                if score.finished:
                    break
            if not score.finished:
                raise ValueError('game exceeded the point limit')
            stage = 'stable' if profile == 'stable' else 'changed' if game_index == 0 else 'restored'
            resume = prefix.with_name(f'{prefix.name}-{stage}-player.json.gz')
            save_player(resume, state, library, player, memory)
            portable = _read(resume)
            portable['motor_guard'] = dict(version=1)
            _write(resume, portable)
            result['games'].append(dict(seed=game_seed, score=score.score, points=points, metrics=_game_summary(points),
                                        active=memory.active, active_kind=memory.maps[memory.active].kind,
                                        checkpoint=_hash(resume)))
            _write(output, result)
            print(stage, score.score, result['games'][-1]['metrics'], flush=True)
    finally:
        arena.close()
    result['replayed_points'] = inspect(_read(output), start)
    _write(output, result)
    print('physical and camera-only replays', result['replayed_points'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('start', type=Path)
    parser.add_argument('prefix', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--perturbation-seed', type=int)
    parser.add_argument('--profile', choices=('coupling', 'gain', 'stable'))
    parser.add_argument('--games', choices=(1, 2), type=int, default=2)
    parser.add_argument('--switch', type=int, default=4)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(inspect(_read(args.prefix.with_suffix('.json.gz')), args.start))
    elif any(v is None for v in (args.seed, args.perturbation_seed, args.profile)):
        parser.error('specify both seeds and the profile')
    else:
        run(args.start, args.prefix, args.seed, args.perturbation_seed, args.profile, args.games, args.switch)
