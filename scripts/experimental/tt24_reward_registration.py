"""Re-register a carried wrist camera online using its history around scalar rewards."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_bounce_ident import WristPupil, _features

DIR = Path('artifacts/tt24')
CHECKPOINT = DIR/'bounce-ident-pupil.npz'
OFFSETS = np.arange(-12, 6)


def survey(arena, seeds, regime, rng):
    examples, shuffled, costs = [], [], []
    for seed in seeds:
        pupil = WristPupil.load(CHECKPOINT)
        obs = arena.reset(seed=seed, split='practice', regime=regime)
        track, paid = [], []
        while True:
            state = pupil.sensor.observe(obs)
            track.append(state[:3] - state[6:9])
            obs, reward, done = arena.step(np.zeros(5))
            if reward > 0:
                paid.append(len(track)-1)
            if done:
                break
        track = np.array(track)
        eligible = [t for t in paid if t+OFFSETS.min() >=0 and t+OFFSETS.max()<len(track)]
        for tick in eligible:
            examples.append(track[tick+OFFSETS])
            arbitrary = rng.integers(-OFFSETS.min(),len(track)-OFFSETS.max())
            shuffled.append(track[arbitrary+OFFSETS])
        costs.append(dict(seed=seed,ticks=len(track),rewards=paid,usable=len(eligible)))
    if not examples:
        raise ValueError('no complete rewarded observation windows')
    return np.median(examples,axis=0), np.median(shuffled,axis=0), costs


def registration(reference, current, affine):
    bias = np.median(reference - current,axis=0)
    if not affine:
        return np.eye(3), np.clip(bias,-.06,.06)
    u,x = current - current.mean(axis=0),reference - reference.mean(axis=0)
    # Sparse, nearly collinear motion leaves weakly observed axes anchored at identity.
    precision = .0015 * np.eye(3)
    transform = np.linalg.solve(u.T@u + precision, u.T@x + precision)
    shift = np.median(reference - current@transform,axis=0)
    return transform, np.clip(shift,-.06,.06)


def episode(arena,seed,transform,shift):
    pupil=WristPupil.load(CHECKPOINT)
    obs=arena.reset(seed=seed,split='held',regime='changed')
    home=obs['joints'].copy()
    tape=[]
    while True:
        state=pupil.sensor.observe(obs)
        state[:3]=state[6:9]+(state[:3]-state[6:9])@transform+shift
        state[3:6]=state[3:6]@transform
        u=pupil.actor.command(_features(state,obs['joints'],home),pupil.issued)
        pupil.issued+=5.2/60*u
        tape.append(u.tolist())
        obs,_,done=arena.step(u)
        if done:
            break
    result=arena.judge_metrics()
    arena.reset(seed=seed,split='held',regime='changed',render=False)
    for u in tape:
        arena.step(u,render=False)
    if arena.judge_metrics()!=result:
        raise ValueError('physical evidence diverged')
    return dict(seed=seed,run=result['longest_consecutive_rebounds'],score=result['rebounds'],ticks=len(tape),commands=tape)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--seed',type=int,default=611451)
    p.add_argument('--count',type=int,default=12)
    p.add_argument('--drops',type=int,default=4)
    p.add_argument('--horizon',type=int,default=450)
    args=p.parse_args()
    rng=np.random.default_rng(args.seed)
    stock=BounceArena(view='wrist',width=192,height=144)
    target=BounceArena(view='wrist',width=192,height=144,arm='alternate',max_steps=args.horizon)
    try:
        ref,shufref,source_cost=survey(stock,rng.integers(2**31-1,size=12).tolist(),'standard',rng)
        current,shuf,target_cost=survey(target,rng.integers(2**31-1,size=args.drops).tolist(),'changed',rng)
        seeds=rng.integers(1,2**31-1,size=args.count).tolist()
        conditions={'frozen':(np.eye(3),np.zeros(3)),'aligned_shift':registration(ref,current,False),
                    'shuffled_shift':registration(shufref,shuf,False),'aligned_affine':registration(ref,current,True)}
        doc=dict(source_cost=source_cost,target_cost=target_cost,held=seeds,conditions={},verified=True)
        for name,(matrix,offset) in conditions.items():
            rows=[episode(target,s,matrix,offset) for s in seeds]
            doc['conditions'][name]=dict(matrix=matrix.tolist(),offset=offset.tolist(),rows=rows)
            print(name,offset,[r['run'] for r in rows],flush=True)
            with gzip.open(args.out,'wt') as f:
                json.dump(doc,f,separators=(',',':'))
    finally:
        stock.close()
        target.close()


if __name__=='__main__':
    main()
