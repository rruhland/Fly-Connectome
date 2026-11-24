"""Export only practice-legal coach traces as raw RGB-derived inputs and motor targets."""

import argparse
from pathlib import Path

import numpy as np

from . import tt33_joint as joint
from .tt4_transfer_trial import _hash, _read, _write
from .tt31_ready import centered
from .tt37_opponent import arena
from .tt37_opponent_benchmark import OUT, PLAN


def export(path, output):
    tape=_read(path)
    lawful={c['key'] for c in _read(PLAN)['practice']}
    if tape['variant']['group']!='practice' or any(c['key'] not in lawful for c in tape['cards']):
        raise ValueError('Do not turn held coach measurements into practice examples')
    arrays={k:[] for k in ('optical','q_rad','velocity_rad_s','target_rad','signed_command','time_index','points')}
    summary=[]
    with centered(tape['ready']):
        a=arena(robust_contact=True,paddle_grip=tape['variant']['grasp'],clearance_margin=.003)
        stereo,rig,*_=joint.resources(a)
        try:
            for i,(c,r) in enumerate(zip(tape['cards'],tape['rows'],strict=True)):
                if not r['score']['learner']:
                    continue
                a.cards=iter([c])
                observed=joint.observe(a,stereo,rig,r,raw=True)
                n=len(observed[0])
                summary.append(dict(id=c['id'],sample_count=n,legal=r['score']['learner']))
                for key,column in zip(arrays,(*observed,np.full(n,i)),strict=True):
                    arrays[key].extend(column)
        finally:
            stereo.close()
            a.close()
    np.savez_compressed(output/'bodycoach-demo.npz',**{k:np.asarray(v) for k,v in arrays.items()})
    _write(output/'bodycoach-demo.json',dict(source_sha256=_hash(path),source=path.name,
        feed_scope='practice only', input_columns=['x','y','z','vx','vy','vz'],
        samples=len(arrays['optical']),points=summary, replay_exact=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('tape',type=Path)
    p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args()
    export(args.tape,args.output)
