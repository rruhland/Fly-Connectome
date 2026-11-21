"""Carry a newly wired arm's table-tennis motor map into a moving-target game."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_memory import load
from .tt11_pursuit_learning import LocalServo
from .tt11_pursuit_trial import fast_play
from .tt11_tracking import TrackingArena
from .tt11_tracking_trial import course, play


def run(starts, output, seed, count):
    entries = [_read(p)['series'] for p in starts]
    if not all(e['arena']==entries[0]['arena'] for e in entries):
        raise ValueError('arm installations must match')
    jacobians = {}
    for label, path, entry in zip(('recalled','two_fresh','commissioned'),starts,entries,strict=True):
        model = load(path)[0].models[entry['model']]
        jacobians[label] = model.jacobian(np.zeros(model.action_size))[:3]
    settings = entries[0]['arena']
    rows = evaluate_maps(jacobians, settings, settings['seed'], seed, count)
    _write(output,dict(seed=seed,starts=[_hash(p) for p in starts],rows=rows,implementation=_hash(Path(__file__))))


def evaluate_maps(jacobians, settings, wiring_seed, seed, count):
    arena = TrackingArena(**dict(settings,render_mode='projected'))
    rows = []
    try:
        for s in np.random.default_rng(seed).integers(0,2**31-1,count).tolist():
            path = course(arena,s)
            for view_name, view in [('familiar',np.eye(3)),('rotated',Rotation.random(random_state=s).as_matrix())]:
                args = arena,wiring_seed,s,path,view
                cases={}
                for label, jac in jacobians.items():
                    result = fast_play(*args,LocalServo(jac,probe=False))
                    if play(*args,tape=result['actions']) != result:
                        raise ValueError('physical moving-target replay diverged')
                    cases[label]=result
                result=fast_play(*args,LocalServo(np.zeros_like(next(iter(jacobians.values()))),probe=True))
                if play(*args,tape=result['actions']) != result:
                    raise ValueError('cold replay diverged')
                cases['target_only']=result
                rows.append(dict(seed=s,view=view_name,rotation=view.tolist(),path=path.tolist(),cases=cases))
    finally:
        arena.close()
    return rows


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('starts',nargs=3,type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--count',type=int,default=8)
    args=p.parse_args()
    run(args.starts,args.output,args.seed,args.count)
