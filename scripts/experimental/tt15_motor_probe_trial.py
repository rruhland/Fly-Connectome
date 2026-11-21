"""Play matched games with a brief optical check after a surprising point."""

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
from .tt15_motor_probe import ProbeMemory, measure


def verify_optical(receipt, source, arena, factory):
    if _hash(source) != receipt['start']:
        raise ValueError('optical replay requires the starting player')
    probes = {}
    for changed in (False, True):
        probes[changed] = wiring_for(arena, receipt, changed)
    initial = receipt['reference']
    if measure(probes[False], initial['seed'], factory) != initial['poses']:
        raise ValueError('reference probe diverged from archive')
    memory = ProbeMemory(memory_from_checkpoint(_read(source), load(source)[0], 'hybrid'), initial['poses'])
    count = 1
    for game in receipt['games']:
        for row in game['points']:
            dry = row['probe']

            def acquire():
                if dry is None:
                    raise ValueError('camera replay expected a recorded physical probe')
                pose = measure(probes[row['changed']], dry['seed'], factory)
                if pose != dry['poses']:
                    raise ValueError('dry response diverged from the optical archive')
                return pose

            trace = [dict(blade=blade, blade_pitch=pitch) for blade, pitch in row['motor_sightings']]
            outcome = memory.observe(row['actions'], trace, acquire)
            if json.loads(json.dumps(outcome)) != row['choice']:
                raise ValueError('repeated optical motor decision diverged')
            count += dry is not None
    return count


def run(start, prefix, seed, perturbation_seed, profile, games, switch):
    arena_settings = _read(start)['series']['arena']
    arena = TT8ServiceArena(**arena_settings)
    actual = perturbation(arena.action_size, perturbation_seed, profile)
    data = dict(schema='tt15-repeat-probe-v1', arena=arena_settings, start=_hash(start), mode='probe',
                seed=seed, profile=profile, perturbation_seed=perturbation_seed, switch=switch,
                matrix=actual.tolist(), games=[], implementation={
                    k: _hash(Path(__file__).with_name(f'tt15_motor_{k}.py')) for k in ('probe', 'probe_trial')})
    output = prefix.with_suffix('.json.gz')
    output.parent.mkdir(exist_ok=True, parents=True)
    resume = start
    try:
        cache_hall_stereo(arena, reflections=True)
        wires = {key: wiring_for(arena, data, key) for key in (False, True)}
        for game_index, game_seed in enumerate(_seeds(seed, 9387, 0, games)['tests']):
            state, library, player = _restore(resume, arena)
            if game_index == 0:
                baseline_seed = _seeds(seed, 77419, 0, 1)['tests'][0]
                baseline = measure(wires[False], baseline_seed, player.tracker_factory)
                data['reference'] = dict(seed=baseline_seed, poses=baseline)
            else:
                baseline = state['motor_probe']['reference']
            memory = ProbeMemory(memory_from_checkpoint(state, library, 'hybrid'), baseline)
            fallback = player._make_servo
            score = ServiceScore(first_server=-1)
            seeds = _seeds(game_seed, 6011, 0, 65)['tests']
            probe_seeds = _seeds(game_seed, 540917, 0, 65)['tests']
            phases = match_phases(PHASES, 65, [game_seed, 6011])
            points = []
            for i, (point_seed, phase, dry_seed) in enumerate(zip(seeds, phases, probe_seeds, strict=True)):
                changed = profile != 'stable' and ((game_index == 0) == (i >= switch))
                motor = memory.maps[memory.active]
                player.adapter = motor.base
                player._make_servo = (lambda: GainServo(motor, player.timing_memory, learn=player.adjust)
                                      ) if motor.coupled else fallback
                row = _point_row(player, arena, wires[changed], game_seed, point_seed, phase, score.server)
                dry = None

                def acquire():
                    nonlocal dry
                    dry = dict(seed=dry_seed, poses=measure(wires[changed], dry_seed, player.tracker_factory))
                    return dry['poses']

                before = process_time_ns()
                decision = memory.observe(row['actions'], player.trace, acquire)
                elapsed = process_time_ns() - before
                row.update(changed=changed, model_used=decision['previous'], model_kind=motor.kind, probe=dry,
                           choice=decision, update_cpu_ns=elapsed,
                           motor_sightings=[(f['blade'], f['blade_pitch']) for f in player.trace])
                points.append(row)
                score.record(row['evaluator'])
                if decision['active'] != decision['previous']:
                    print(game_index + 1, i + 1, decision, flush=True)
                if score.finished:
                    break
            if not score.finished:
                raise ValueError('game exceeded the point limit')
            stage = 'stable' if profile == 'stable' else 'changed' if game_index == 0 else 'restored'
            resume = prefix.with_name(f'{prefix.name}-{stage}-player.json.gz')
            save_player(resume, state, library, player, memory)
            portable = _read(resume)
            portable['motor_probe'] = dict(reference=baseline, version=1)
            _write(resume, portable)
            data['games'].append(dict(seed=game_seed, score=score.score, points=points, metrics=_game_summary(points),
                                      active=memory.active, active_kind=memory.maps[memory.active].kind,
                                      checkpoint=_hash(resume)))
            _write(output, data)
            print(stage, score.score, sum(row['probe'] is not None for row in points), flush=True)
        archived = _read(output)
        data['replayed_probes'] = verify_optical(archived, start, arena, player.tracker_factory)
    finally:
        arena.close()
    data['replayed_points'] = verify(archived)
    _write(output, data)
    print('physical point and optical probe replays', data['replayed_points'], data['replayed_probes'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('start', type=Path)
    p.add_argument('prefix', type=Path)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--perturbation-seed', type=int, required=True)
    p.add_argument('--profile', choices=('coupling', 'gain', 'stable'), required=True)
    p.add_argument('--games', choices=(1, 2), type=int, default=2)
    p.add_argument('--switch', type=int, default=4)
    a = p.parse_args()
    run(a.start, a.prefix, a.seed, a.perturbation_seed, a.profile, a.games, a.switch)
