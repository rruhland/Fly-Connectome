"""Adapt the strength of transferred action effects with a small number of scored drops."""
import argparse
import gzip
import json
import numpy as np
from .tt24_bounce import BounceArena
from .tt24_world_arm import DIR, Grounding, perception
from .tt24_world_prior import WorldPrior


def play(arena,prior,seed,weights,split,permute,verify):
    sensor=perception(False,True)
    obs=arena.reset(seed=seed,split=split)
    body=Grounding(sensor.decoder,obs['joints'])
    neutral=np.r_[np.zeros(3),body.origin[3:]]
    tape=[]
    while True:
        state=sensor.observe(obs)
        state[9:11]-=neutral[3:]
        effect=prior.target(state)
        if permute:
            units=np.array((.06,.06,.06,.30,.30))
            effect=units*(effect/units)[[2,4,0,1,3]]
        effect=neutral+np.r_[weights[:1].repeat(2),weights[1],weights[2:].repeat(2)]*effect
        u=body.target(effect)
        tape.append(u.tolist())
        obs,_,done=arena.step(u)
        if done:break
    judge=arena.judge_metrics()
    if verify:
        arena.reset(seed=seed,split=split,render=False)
        for u in tape:
            arena.step(u,render=False)
        if arena.judge_metrics()!=judge:
            raise ValueError('held motor tape failed')
    return dict(run=judge['longest_consecutive_rebounds'],score=judge['rebounds'],ticks=len(tape),actions=tape if verify else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=str)
    p.add_argument('--source',default='world-prior-body.npz')
    p.add_argument('--streams',type=int,default=3)
    p.add_argument('--count',type=int,default=12)
    args=p.parse_args()
    prior=WorldPrior.load(DIR/args.source)
    rng=np.random.default_rng(67129417)
    # Generic translation, vertical and rotation gains. No piece of motion is replayed.
    gains=np.array(((0.,0.,0.),(.25,.25,.25),(.55,.55,.55),(0.,.50,0.),(.35,0.,.35),(.15,.70,.08)))
    arena=BounceArena(view='wrist',width=192,height=144,max_steps=360)
    streams=[]
    try:
        for i in range(args.streams):
            practice=rng.integers(1,2**31-1,size=2).tolist()
            starts=rng.integers(1,2**31-1,size=args.count).tolist()
            result={}
            for mode in ('coherent','scrambled'):
                permute=mode=='scrambled'
                rows=[[play(arena,prior,s,g,'practice',permute,False) for s in practice] for g in gains]
                value=[sum(r['score']+.02*r['ticks']/60 for r in group) for group in rows]
                best=int(np.argmax(value))
                held=[play(arena,prior,s,gains[best],'held',permute,True) for s in starts]
                result[mode]=dict(practice=rows,chosen=best,held=held)
            control=[play(arena,prior,s,gains[0],'held',False,True) for s in starts]
            print(i,[(k,v['chosen'],sum(r['run'] for r in v['held'])) for k,v in result.items()],sum(r['run']for r in control),flush=True)
            streams.append(dict(practice_seeds=practice,held_seeds=starts,conditions=result,anchor=control))
            with gzip.open(args.out,'wt') as f:
                json.dump(dict(source=args.source,gains=gains.tolist(),streams=streams,verified=True),f,separators=(',',':'))
    finally:
        arena.close()


if __name__=='__main__':
    main()
