"""Recall learned wrist calibrations from appearance while carrying one joint policy."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import sobel

from .tt24_bounce import BounceArena
from .tt24_bounce_ident import WristPupil, _ellipse, _features, _INC
from .tt24_metric_transfer import load_transferred

DIR=Path('artifacts/tt24')
BASE=DIR/'bounce-ident-pupil.npz'
CORRECTION=DIR/'metric-transfer-map.npz'


def descriptor(rgb):
    # Standardized coarse intensity and edge energy from the whole current camera.
    gray=rgb.astype(float).mean(axis=2)/255.
    edge=np.hypot(sobel(gray,axis=0),sobel(gray,axis=1))
    features=[]
    for channel in (gray,edge):
        grid=channel.reshape(12,12,16,12).mean(axis=(1,3)).ravel()
        grid-=grid.mean()
        features.append(grid/(np.linalg.norm(grid)+1.e-8))
    return np.concatenate(features)


def memory(arenas,seeds):
    return np.asarray([[descriptor(arena.reset(seed=s,split='practice',
        regime='changed' if index else 'standard')['rgb']) for s in seeds[index]]
        for index,arena in enumerate(arenas)])


def evaluate(arena,seed,changed,prototypes,mode):
    stock=WristPupil.load(BASE)
    other=load_transferred(BASE,CORRECTION)
    for field in ('weights','scale','center'):
        if not np.array_equal(getattr(stock.actor,field),getattr(other.actor,field)):
            raise ValueError('action readouts differ')
    obs=arena.reset(seed=seed,split='held',regime='changed' if changed else 'standard')
    home=obs['joints'].copy()
    chosen=[]
    tape=[]
    qvals=None
    while True:
        tick=len(tape)
        if tick%6==0:
            raw=np.max(prototypes@descriptor(obs['rgb']),axis=1)
            qvals=raw if qvals is None else .3*qvals+.7*raw
        weight=float(np.argmax(qvals)) if mode=='recall' else {'stock':0.,'recent':1.,'average':.5}[mode]
        chosen.append(weight)
        ellipse=_ellipse(obs['rgb'])
        a=stock.sensor.read(ellipse,obs['joints'])
        b=other.sensor.read(ellipse,obs['joints'])
        state=a+weight*(b-a) if weight!=1. else b
        u=stock.actor.command(_features(state,obs['joints'],home),stock.issued)
        stock.issued+=_INC*u
        tape.append(u.tolist())
        obs,_,done=arena.step(u)
        if done:
            break
    judge=arena.judge_metrics()
    arena.reset(seed=seed,split='held',regime='changed' if changed else 'standard',render=False)
    for u in tape:
        arena.step(u,render=False)
    if arena.judge_metrics()!=judge:
        raise ValueError('replay diverged')
    return dict(seed=seed,run=judge['longest_consecutive_rebounds'],score=judge['rebounds'],actions=tape,
                choices=chosen,initial=chosen[0],fraction_target=sum(chosen)/len(chosen))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--seed',type=int,default=41302129)
    p.add_argument('--drops',type=int,default=24)
    args=p.parse_args()
    arenas=[BounceArena(view='wrist',width=192,height=144,arm=arm,max_steps=450) for arm in ('stock','alternate')]
    rng=np.random.default_rng(args.seed)
    try:
        template_seeds=rng.integers(1,2**31-1,(2,4)).tolist()
        bank=memory(arenas,template_seeds)
        starts=rng.integers(1,2**31-1,args.drops).tolist()
        order=np.tile([0,0,1,1,1,0],1+len(starts)//6)[:len(starts)].tolist()
        doc=dict(colors='grayscale and edge features',seeds=starts,changed=order,reference_images=template_seeds,
                 source=BASE.name,correction=CORRECTION.name,verified=True,modes={})
        for mode in ('recall','stock','recent','average'):
            rows=[evaluate(arenas[idx],s,idx,bank,mode) for idx,s in zip(order,starts,strict=True)]
            doc['modes'][mode]=rows
            print(mode,[(sum(r['run'] for r,idx in zip(rows,order) if idx==j),
                         sum(r['initial']==j for r,idx in zip(rows,order) if idx==j)) for j in (0,1)],flush=True)
            with gzip.open(args.out,'wt') as f:
                json.dump(doc,f,separators=(',',':'))
    finally:
        for arena in arenas:
            arena.close()


if __name__=='__main__':
    main()
