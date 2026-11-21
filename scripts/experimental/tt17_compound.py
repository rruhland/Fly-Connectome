"""Separate a motor reassignment from cross-axis coupling with two RGB movements."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _seeds, _write
from .tt6_body_probe import collect_probe
from .tt6_continuous_trial import match_phases
from .tt6_render import cache_hall_stereo
from .tt10_series_trial import PHASES
from .tt10_target import TargetArena
from .tt12_motor_recall import rewire
from .tt14_gain_hardware import GainWiring, perturbation
from .tt14_gain_model import MotorMap
from .tt14_motor_audit import displacement, move
from .tt16_coded_probe import decode
from .tt17_motor_fit import initialise
from .tt17_motor_protocol import load_optical_player, make_arena, observed, setup_path, wiring_for


def permutation_matrix(order, signs):
    matrix = np.zeros((len(order), len(order)))
    matrix[order, np.arange(len(order))] = signs
    return matrix


def path_for(folder, case):
    return folder / f'compound-{case}-query.json.gz'


def prepare(folder, case):
    site, index = case.split('-')
    index = int(index)
    parent = setup_path(folder.parent, f'{site}-coupling')
    spec = _read(parent)
    rng = np.random.default_rng([61732859, int(site=='five'), index])
    width = len(spec['matrix'])
    order = rng.permutation(width).tolist()
    signs = rng.choice([-1, 1], size=width).tolist()
    permutation = permutation_matrix(order,signs)
    mixer_seed, game_seed, first_scene, second_scene, pulse_seed = rng.integers(0, 2**31-1, size=5).tolist()
    matrix = perturbation(width,mixer_seed,'coupling') @ permutation
    spec.update(case=case, matrix=matrix.tolist(), parent=_hash(parent), game_seed=game_seed,
                starts=_seeds(game_seed, 6011,0,65)['tests'],
                phases=match_phases(PHASES,65,[game_seed,6011]),
                followups=_seeds(game_seed,540917,0,65)['tests'])
    actions = np.array(spec['actions'])
    native = actions[:12]
    arena = make_arena(spec)
    try:
        cache_hall_stereo(arena,reflections=True)
        _, _, player, archived = load_optical_player(spec,arena)
        source = archived.source
        wire = wiring_for(arena,spec)
        raw = collect_probe(wire,first_scene,actions)
        if len(raw) != 25:
            raise ValueError('continuous probe ended early')
        spec['active'] = observed(raw,player.tracker_factory)
        early = observed(raw[:13],player.tracker_factory)
        first_key, _ = decode(source,native,early['ticks'],early['seen'])
        change_basis = permutation_matrix(*first_key)
        # In row convention a desired original motion is requested on the
        # new controls by multiplication with the estimated permutation.
        remapped_commands = native @ change_basis
        second = collect_probe(wire,second_scene,remapped_commands)
        optical = observed(second,player.tracker_factory)
        if len(second)!=13 or optical != observed(collect_probe(wire,second_scene,remapped_commands),
                                                  player.tracker_factory):
            raise ValueError('recoded movement did not replay')
        first_sightings = [i for i,t in enumerate(spec['reference']['ticks']) if t<=12]
        reference = {k:[spec['reference'][k][i] for i in first_sightings] for k in ('ticks','seen')}
        query = dict(actions=native.tolist(), early=early, first_key=first_key, second=optical, reference=reference,
                     second_actions=remapped_commands.tolist(), scenes=[first_scene, second_scene],
                     evaluator_key=[order,signs], pulse_seed=pulse_seed)
    finally:
        arena.close()
    folder.mkdir(parents=True,exist_ok=True)
    _write(setup_path(folder,case),spec)
    _write(path_for(folder,case),query)


def recoded(source, query):
    commands = np.array(query['actions'])
    early = query['early']
    key, _ = decode(source,commands,early['ticks'],early['seen'])
    change_basis = permutation_matrix(*key)
    if key!=query['first_key'] or not np.array_equal(commands @ change_basis, query['second_actions']):
        raise ValueError('second query does not follow the first optical decision')
    # The second fit sees exactly the original requested movement in these
    # coordinates. Its residual mixing is then expressed in public controls.
    _, corrected, _, evidence = initialise(source,commands,query['reference'],query['second'])
    base = rewire(corrected.base, *key)
    motor = MotorMap(base,change_basis.T @ corrected.transform @ change_basis,
                     corrected.kind if corrected.coupled else 'remap')
    return motor, dict(evidence, first_key=key, second_key=evidence['signed'])


def held_motion(folder, case):
    spec = _read(setup_path(folder,case))
    query = _read(path_for(folder,case))
    arena = make_arena(spec)
    try:
        _,_,_,archived = load_optical_player(spec,arena)
        source = archived.source
    finally:
        arena.close()
    signed, bounded, _, _ = initialise(source,spec['actions'],spec['reference'],spec['active'])
    revised, evidence = recoded(source,query)
    arena = TargetArena(**dict(spec['arena'],render_mode='projected'))
    rows = []
    try:
        native = GainWiring(arena,spec['wiring'])
        changed = wiring_for(arena,spec)
        for seed in np.random.default_rng(query['pulse_seed']).integers(0,2**31-1,12).tolist():
            actions = np.zeros((56,source.action_size))
            random = np.random.default_rng(seed)
            for tick in (0,14,28,42):
                actions[tick:tick+3] = random.uniform(-.42,.42,source.action_size)
            original = move(native,seed,actions)
            new = move(changed,seed,actions)
            template = displacement(source,np.eye(source.action_size),actions)
            errors = {}
            for name, motor in [('signed',signed),('bounded',bounded),('recoded',revised)]:
                predicted = displacement(motor.base,motor.transform,actions)
                errors[name] = float(np.linalg.norm(predicted-template-new+original,axis=1).mean())
            rows.append(dict(seed=seed,errors_m=errors))
    finally:
        arena.close()
    _write(folder/f'compound-{case}-motion.json.gz',dict(query_sha256=_hash(path_for(folder,case)),
           setup_sha256=_hash(setup_path(folder,case)),recoded=evidence,held=rows))
    print(case,query['first_key']==query['evaluator_key'],evidence['reason'],
          {key:round(1000*np.mean([r['errors_m'][key] for r in rows])) for key in rows[0]['errors_m']},flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('case',choices=('gantry-0','gantry-1','five-0','five-1'))
    a=parser.parse_args()
    prepare(a.folder,a.case)
    held_motion(a.folder,a.case)
