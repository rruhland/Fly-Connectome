"""Check if a carried spatial preference helps on unassisted, off-center SO-101 openings."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt22_joint_practice import draw, point
from .tt22_motion_prefix import prefix
from .tt22_motor_transfer_trial import MotorArena


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=652338042)
    p.add_argument('--count',type=int,default=64)
    args=p.parse_args()
    source=_read(args.source)
    if not source.get('generic'):
        raise ValueError('expected the raw stereo motion task')
    rng=np.random.default_rng(args.seed)
    held=draw(rng,args.count,(.60,1.0))
    arena=MotorArena()
    template=np.array(source['template'])
    data=dict(seed=args.seed,optics=[],scored=[],replicates=[])
    cache={}
    try:
        data['optics']=[prefix(arena,s,lane) for s,lane in held]
        for rep in source['replicates']:
            record={}
            for name,snapshots in rep['eval'].items():
                record[name]=[]
                for chk in snapshots:
                    key=tuple(chk['params'])
                    if key not in cache:
                        rows=[point(arena,template,key,obs) for obs in data['optics']]
                        cache[key]=len(data['scored'])
                        data['scored'].append(dict(params=key,rows=rows))
                    record[name].append(dict(budget=chk['budget'],index=cache[key]))
            data['replicates'].append(record)
            _write(args.output,data)
            print('evaluated stream',len(data['replicates']),flush=True)
    finally:
        arena.close()


if __name__=='__main__':
    main()
