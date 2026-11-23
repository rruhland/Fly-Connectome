"""Transfer motor coordination from actions actually executed by a table-tennis pupil."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_bounce_trial import play, search, verify_replay
from .tt24_motor import MotorEffects


DIR=Path('artifacts/tt24')


def motor_modes(episodes, shuffled):
    rng=np.random.default_rng(631890)
    cov=[]
    for a in episodes:
        # Filter encoder-scale motor jitter before measuring joint co-activation.
        kernel=np.ones(4)/4
        smooth=np.column_stack([np.convolve(u,kernel,'valid') for u in a.T])
        if shuffled:
            smooth=np.column_stack([np.roll(u,rng.integers(len(u))) for u in smooth.T])
        cov.append(smooth.T@smooth/len(smooth))
    matrix=np.mean(cov,axis=0)
    matrix=.88*matrix+.12*np.eye(5)*np.trace(matrix)/5.
    eig,basis=np.linalg.eigh(matrix/(np.trace(matrix)/5.))
    return (basis*np.sqrt(np.maximum(eig,1e-6)))@basis.T


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--streams',type=int,default=4)
    p.add_argument('--budget',type=int,default=24)
    args=p.parse_args()
    with gzip.open(DIR/'rally-bank-held.json.gz','rt') as f:
        source=json.load(f)
    episodes=[np.array(c['results']['broad']['actions'])[:,:5] for c in source['cases']]
    modes=dict(table=motor_modes(episodes,False), shuffled=motor_modes(episodes,True), cold=np.eye(5))
    rng=np.random.default_rng(96845551)
    allruns=[]
    arena=BounceArena(view='wrist',width=64,height=48)
    try:
        for _ in range(args.streams):
            seed=int(rng.integers(2**31-1))
            practice=rng.integers(1,2**31-1,4).tolist()
            schedule=(practice*(1+args.budget//4))[:args.budget]
            held=rng.integers(1,2**31-1,12).tolist()
            records={}
            for name,m in modes.items():
                eff=MotorEffects(5)
                w,training=search(arena,schedule,None,eff,seed=seed,radius=1.1,motor_modes=m,
                                  readout='rate',repeats=2)
                ss=[]
                for s in held:
                    row=play(arena,s,w,eff,split='held',readout='rate',stride=2,keep_actions=True)
                    verify_replay(arena,row)
                    ss.append(dict(seed=s,n=row['rebounds'],run=row['longest_consecutive_rebounds']))
                records[name]=dict(weights=w.tolist(),score=ss,practice=[r['rebounds'] for r in training])
            hold=[play(arena,s,w,eff,split='held',hold=True,blind=True)['rebounds'] for s in held]
            print({k:sum(r['n'] for r in v['score']) for k,v in records.items()},sum(hold),flush=True)
            allruns.append(dict(practice=schedule,seed=seed,held=held,score=records,hold=hold))
        with gzip.open(args.out,'wt') as f:
            json.dump(dict(source_ticks=sum(map(len,episodes)),motor_bases={k:v.tolist() for k,v in modes.items()},
                           trials=allruns),f)
    finally:
        arena.close()


if __name__=='__main__':
    main()
