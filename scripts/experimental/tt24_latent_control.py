"""Learn direct wrist control with an angular-size cue and interactive demonstrations.

A camera-relative representation uses the inverse angular extent of a tracked
region. No camera or object dimensions, joint kinematics, flight model, or
impact timer are supplied. All student actions are continuous joint rates.
"""
import argparse
from collections import deque
import gzip
import json
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_motor import optical_sightings


class Sensor:
    def __init__(self, obs):
        self.home = obs['joints'].copy()
        self.target = np.zeros(5)
        self.history = deque(maxlen=8)
        self.age = 10
        self.last = np.zeros(3)
        self.initial = None

    def read(self, obs, *, blind=False):
        mover, _ = (None,None) if blind else optical_sightings(obs['rgb'])
        if mover:
            extent = max(.018, mover.radius)
            self.last = np.r_[mover.center, 1.] * (.08 / extent)
            self.age = 0
        else:
            self.age = min(self.age+1,10)
        if self.initial is None and mover:
            self.initial = self.last.copy()
        self.history.append(self.last)
        hist = np.array(self.history)
        short = (hist[-1]-hist[max(0,len(hist)-3)]) / min(2,max(1,len(hist)-1))
        long = (hist[-1]-hist[0])/max(1,len(hist)-1)
        q = obs['joints']-self.home
        return np.r_[self.last, short, long, self.age*.1,
                     q, obs['joint_velocity'], self.target, self.home,
                     self.initial if self.initial is not None else np.zeros(3)]


class Readout:
    def __init__(self, x=None, y=None, *, checkpoint=None):
        if checkpoint is not None:
            self.__dict__.update(checkpoint)
            return
        self.center=x.mean(0)
        self.scale=np.maximum(x.std(0), np.r_[[.025]*3, [.012]*6,.12,
                                             [.025]*5,[.22]*5,[.025]*5,[.05]*5,[.1]*3])
        rng=np.random.default_rng(327912)
        self.proj=rng.normal(size=(x.shape[1],512)) / np.sqrt(x.shape[1])
        self.proj *= rng.uniform(.55,2.4,self.proj.shape[1])
        self.phase=rng.uniform(-np.pi,np.pi,self.proj.shape[1])
        design=self.design(x)
        # Global ridge interpolation smooths fluctuations in the teacher's servo commands.
        self.coef=np.linalg.solve(design.T@design + .22*np.eye(design.shape[1]), design.T@y)

    def design(self, x):
        x=np.clip((np.atleast_2d(x)-self.center)/self.scale,-5,5)
        return np.column_stack((np.ones(len(x)),x,np.cos(x@self.proj+self.phase)))

    def command(self, x, target, dt):
        offset, rate=np.split((self.design(x)@self.coef)[0],2)
        return np.clip(.85*(offset-target)/(5.2*dt) + .15*rate, -.9,.9)

    def save(self, path):
        np.savez_compressed(path,**self.__dict__)


def collect(arena, seed, actor, *, teacher_fraction=0., labels=False, split='held', regime='standard',
            blind=False, collect_images=False):
    obs=arena.reset(seed=seed,split=split,regime=regime,render=not blind)
    sense=Sensor(obs)
    if labels:
        from .tt24_bounce_ceiling import PrivilegedCeiling
        tutor=PrivilegedCeiling(arena)
    xx,yy=[],[]
    n=0
    done=False
    while not done:
        x=sense.read(obs,blind=blind)
        command=actor.command(x,sense.target,arena.dt) if actor else np.zeros(5)
        if labels:
            expert=tutor.command()
            xx.append(x)
            yy.append(np.r_[sense.target+expert*(5.2*arena.dt),expert])
            command=teacher_fraction*expert+(1-teacher_fraction)*command
        obs,_,done=arena.step(command,render=not blind)
        sense.target += command*(5.2*arena.dt)
        n+=1
    judge=arena.judge_metrics()
    return dict(seed=seed,total=judge['rebounds'],consecutive=judge['longest_consecutive_rebounds'],frames=n), (xx,yy)


def pack(groups):
    return tuple(np.concatenate(c) for c in zip(*groups))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--examples',type=Path)
    p.add_argument('--seed',type=int,default=236516)
    p.add_argument('--demos',type=int,default=8)
    p.add_argument('--blocks',type=int,default=3)
    p.add_argument('--per-block',type=int,default=4)
    p.add_argument('--held',type=int,default=12)
    p.add_argument('--arm',default='stock')
    p.add_argument('--regime',default='standard')
    p.add_argument('--prior',type=Path)
    args=p.parse_args()
    env=BounceArena(view='wrist',width=96,height=72,arm=args.arm)
    rng=np.random.default_rng(args.seed)
    prior=[]
    if args.prior:
        with np.load(args.prior) as cache:
            prior=[(cache['x'],cache['y'])]
    groups=list(prior)
    exposure=[]
    result={}
    held=rng.integers(1,2**31-1,args.held).tolist()
    schedule=[(args.demos,1.)]+[(args.per_block,x) for x in [.45,.18,0.][:args.blocks]]
    actor=Readout(*pack(groups)) if groups else None
    try:
        for index, (count,weight) in enumerate(schedule):
            for seed in rng.integers(1,2**31-1,count).tolist():
                r,data=collect(env,seed,actor,teacher_fraction=weight,labels=True,split='practice',regime=args.regime)
                groups.append(data)
                exposure.append(r | dict(teacher_fraction=weight))
            if groups:
                actor=Readout(*pack(groups))
            ss=[collect(env,s,actor,regime=args.regime)[0] for s in held]
            result[str(index)]=ss
            print('stage',index,[x['consecutive'] for x in ss],flush=True)
        result['blind']=[collect(env,s,actor,blind=True,regime=args.regime)[0] for s in held]
        result['hold']=[collect(env,s,None,blind=True,regime=args.regime)[0] for s in held]
        actor.save(args.checkpoint)
        if args.examples:
            x,y=pack(groups)
            np.savez_compressed(args.examples,x=x,y=y)
        with gzip.open(args.output,'wt') as f:
            json.dump(dict(seed=args.seed,arm=args.arm,regime=args.regime,held=held,exposure=exposure,
                           source_examples=sum(len(g[0]) for g in prior),results=result),f)
    finally:
        env.close()


if __name__=='__main__':
    main()
