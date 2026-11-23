"""Refit a direct motor readout on the discrepancy between practiced and deployed senses."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_bounce_ident import WristPupil, _training_demos, wrist_features, _save_pupil
from .tt24_metric_transfer import load_transferred

DIR=Path('artifacts/tt24')
SOURCE=DIR/'bounce-ident-pupil.npz'
PUBLIC=DIR/'metric-transfer-map.npz'


def cache():
    arena=BounceArena(view='wrist',width=192,height=144)
    try:
        _,pairs,records=_training_demos(arena)
    finally:
        arena.close()
    source=WristPupil.load(SOURCE)
    observed=wrist_features(source.sensor.decoder,source.sensor.motion,records)
    offsets=np.cumsum([0]+list(map(len,observed)))
    np.savez_compressed(DIR/'sensor-align-examples.npz',clean=np.concatenate([p[0] for p in pairs]),
                        observed=np.concatenate(observed),labels=np.concatenate([p[1] for p in pairs]),offsets=offsets)


def fit(mode, file):
    agent=WristPupil.load(SOURCE)
    clean,x,y=(file[k] for k in ('clean','observed','labels'))
    # Fix the basis and the original feature selection across comparisons.
    if mode=='native':
        return agent
    if mode=='sensor':
        data,target=x,y
    elif mode=='joint':
        data=np.concatenate((clean,x,(clean+x)/2),axis=0)
        target=np.tile(y,(3,1))
    elif mode=='jitter':
        rng=np.random.default_rng(1102983)
        errors=x-clean
        samples=[clean,x]
        for _ in range(3):
            ids=rng.permutation(len(x))
            # A joint-vector error preserves inter-feature covariance.
            samples.append(clean+errors[ids]*rng.uniform(.2,1.2,(len(x),1)))
        data=np.concatenate(samples)
        target=np.tile(y,(len(samples),1))
    else:
        raise ValueError(mode)
    phi=agent.actor.design(data)
    gram=phi.T@phi
    initial=agent.actor.weights
    # Small residual prior keeps unsupported feature directions at their learned behavior.
    ridge=.5
    residual=np.linalg.solve(gram+ridge*np.eye(gram.shape[0]),phi.T@(target-phi@initial))
    agent.actor.weights=initial+residual
    return agent


def run(arena,seed,parameters,regime,confirm):
    pupil=WristPupil.load(SOURCE) if regime=='standard' else load_transferred(SOURCE,PUBLIC)
    pupil.actor.weights=parameters
    obs=arena.reset(seed=seed,split='held',regime=regime)
    tape=[]
    while True:
        u=pupil.command(obs)
        tape.append(u.tolist())
        obs,_,done=arena.step(u)
        if done:break
    judge=arena.judge_metrics()
    if confirm:
        arena.reset(seed=seed,split='held',regime=regime,render=False)
        for u in tape:
            arena.step(u,render=False)
        if arena.judge_metrics()!=judge:
            raise ValueError('tape diverged')
    return dict(run=judge['longest_consecutive_rebounds'],score=judge['rebounds'],ticks=len(tape),
                actions=tape if confirm else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--arm',default='stock',choices=['stock','alternate'])
    p.add_argument('--modes',nargs='+',default=['native','sensor','joint','jitter'])
    p.add_argument('--seed',type=int,default=95103818)
    p.add_argument('--count',type=int,default=8)
    p.add_argument('--horizon',type=int,default=300)
    p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    if not (DIR/'sensor-align-examples.npz').exists():
        cache()
    seeds=np.random.default_rng(args.seed).integers(1,2**31-1,args.count).tolist()
    arena=BounceArena(view='wrist',width=192,height=144,arm=args.arm,max_steps=args.horizon)
    regime='standard' if args.arm=='stock' else 'changed'
    try:
        doc=dict(arm=args.arm,seeds=seeds,horizon=args.horizon,verified=args.verify,results={})
        with np.load(DIR/'sensor-align-examples.npz') as data:
            for mode in args.modes:
                pupil=fit(mode,data)
                if mode!='native':
                    _save_pupil(DIR/f'sensor-align-{mode}.npz',pupil.actor,pupil.sensor.decoder,pupil.sensor.motion)
                rows=[run(arena,s,pupil.actor.weights,regime,args.verify) for s in seeds]
                print(mode,[r['run'] for r in rows],flush=True)
                doc['results'][mode]=rows
                with gzip.open(args.out,'wt') as f:
                    json.dump(doc,f,separators=(',',':'))
    finally:
        arena.close()


if __name__=='__main__':
    main()
