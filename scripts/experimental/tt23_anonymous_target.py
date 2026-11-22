"""Test Pong-learned discrete jogs on anonymous camera observations at the arm."""

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_duration_transfer import MACRO, STREAMS
from .tt22_joint_practice import READS, draw
from .tt22_motion_prefix import prefix
from .tt22_motor_transfer import MotorPrimitive
from .tt22_motor_transfer_trial import MotorArena
from .tt23_anonymous_action import CHECK, OPTIONS, features, proposal
from .tt23_optical_contact import OUT


def target_context(sight,ready):
    if sight['mover'] is None:
        return None
    mover,speed=np.asarray(sight['mover'])
    dims=np.array([.848,1.60])
    return features((mover[:2]-ready[:2])/dims,speed[:2]/dims)


def run(seed=193856087, output=OUT/"anonymous-action-arm.json.gz"):
    source=_read(CHECK)
    macro=np.asarray(_read(MACRO)['parameters'])
    seed_generator=seed
    order=list(STREAMS)
    rng=np.random.default_rng(seed)
    batches=[draw(rng,32,STREAMS[k]) for k in order]
    points=[(name,*point) for pair in zip(*batches,strict=True) for name,point in zip(order,pair,strict=True)]
    arena=MotorArena()
    rows=[]
    try:
        for domain,seed,lane in points:
            sight=prefix(arena,seed,lane)
            pose=MotorPrimitive(macro,None).ready_world
            x=target_context(sight,pose)
            # Every option is evaluated physically. Learning or inference gets only the chosen option.
            outcomes=[]
            for jog in OPTIONS:
                arena.lane=lane
                arena.reset(seed=seed,server=-1,render=False)
                move=MotorPrimitive(macro,None)
                if x is not None:
                    move._translate(pose[0]+jog*.848,pose)
                for tick in range(arena.max_steps):
                    a=np.zeros(6) if tick<READS else move.command(tick)
                    _,_,done,cap,info=arena.step(a,render=False)
                    if done or cap:
                        break
                outcomes.append(dict(ticks=arena.steps,contacts=info['contacts'],legal=info['legal_returns'],fault=info['reason']))
            choices={'learned':3 if x is None else int(np.argmax(proposal(source['model'],x))),
                     'shuffled':3 if x is None else int(np.argmax(proposal(source['shuffled'],x))),'neutral':3}
            rows.append(dict(domain=domain,seed=seed,lane=lane,sight=sight,options=outcomes,choices=choices))
            if len(rows)%8==0:
                print('covered',len(rows),flush=True)
    finally:
        arena.close()
    summary={k:{field:sum(r['options'][r['choices'][k]][field]>0 for r in rows) for field in ('contacts','legal')}
                  for k in ('learned','shuffled','neutral')}
    _write(output,dict(seed_generator=seed_generator,source_hash=_hash(CHECK),macro_hash=_hash(MACRO),rows=rows,summary=summary))
    print(summary,flush=True)


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=193856087)
    parser.add_argument("--output",type=Path,default=OUT/"anonymous-action-arm.json.gz")
    args=parser.parse_args()
    run(args.seed,args.output)
