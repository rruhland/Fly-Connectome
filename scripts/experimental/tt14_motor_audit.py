"""Evaluator-only generalization to new, identical actuator command sequences."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_timing import DelayedWiring, Disturbance
from .tt10_target import TargetArena
from .tt13_live_ports import effective
from .tt14_gain_hardware import GainWiring
from .tt14_gain_trial import load_map


def displacement(model, transform, actions):
    requested = np.vstack([np.zeros(model.action_size),actions]) @ transform.T
    positions = model.pose(effective(requested,model.kernels))[:,:3]
    return positions[1:] - positions[0]


def move(wiring, seed, actions):
    arena = wiring.arena
    noise = Disturbance(delays=((0,0),(19,2),(39,1)),noise=((0,0.),(19,.025)))
    hardware = DelayedWiring(wiring,noise,seed=seed)
    hardware.reset(seed=seed, render=False)
    start = arena.paddle.copy()
    path = []
    for action in actions:
        hardware.step(action,render=False)
        path.append(arena.paddle-start)
    return np.array(path)


def run(folder, site, seed, count):
    models, sources = {}, {}
    for name in ('fixed','refit','remap','hybrid'):
        prefix = folder/f'gain-{site}-coupling-{name}'
        checkpoint = prefix.with_name(prefix.name+'-changed-player.json.gz')
        table = _read(prefix.with_suffix('.json.gz'))
        sources[name] = _hash(checkpoint)
        if sources[name] != table['games'][0]['checkpoint']:
            raise ValueError('different first game and checkpoint')
        models[name] = load_map(checkpoint)
    settings = table['arena']
    arena = TargetArena(**dict(settings,render_mode='projected'))
    true_mix = np.asarray(table['matrix'])
    native = np.eye(arena.action_size)
    models['known_mixer'] = (models['fixed'][0], true_mix)
    # A point reset replaces arena.seed. Build both harnesses from the fixed
    # installation seed before any trial changes the environment's episode seed.
    native_wiring = GainWiring(arena, settings['seed'], native)
    changed_wiring = GainWiring(arena, settings['seed'], true_mix)
    rows = []
    try:
        for s in np.random.default_rng(seed).integers(0,2**31-1,count).tolist():
            rng = np.random.default_rng(s)
            actions = np.zeros((56,arena.action_size))
            for tick in (0,14,28,42):
                actions[tick:tick+3] = rng.uniform(-.42,.42,arena.action_size)
            original = move(native_wiring,s,actions)
            changed = move(changed_wiring,s,actions)
            if not (np.array_equal(changed,move(changed_wiring,s,actions)) and
                    np.array_equal(original,move(native_wiring,s,actions))):
                raise ValueError('physical endpoint replay differed')
            reference = displacement(*models['fixed'],actions)
            errors = {}
            for name, (model,mix) in models.items():
                prediction = displacement(model,mix,actions)
                absolute = np.linalg.norm(prediction-changed,axis=1)
                relative = np.linalg.norm((prediction-reference)-(changed-original),axis=1)
                errors[name] = dict(mean_m=absolute.mean(),change_m=relative.mean(),
                                    settled_m=absolute[13::14].mean(),settled_change_m=relative[13::14].mean())
            rows.append(dict(seed=s,actions=actions.tolist(),original=original.tolist(),changed=changed.tolist(),
                             errors=errors))
    finally:
        arena.close()
    _write(folder/f'motor-audit-{site}.json.gz',dict(seed=seed,checkpoints=sources,rows=rows,
                                                    implementation=_hash(Path(__file__))))
    print({k:round(np.mean([r['errors'][k]['change_m'] for r in rows])*1000,1) for k in models})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('site',choices=('gantry','five'))
    p.add_argument('--seed',required=True,type=int)
    p.add_argument('--count',default=24,type=int)
    a=p.parse_args()
    run(a.folder,a.site,a.seed,a.count)
