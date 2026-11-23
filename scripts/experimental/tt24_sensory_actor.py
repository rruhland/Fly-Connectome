"""Observation-conditioned joint feedback with bounded interactive instruction.

The student sees optical angular position/extent, their history, encoders and
its own commands. It fits local linear readouts with no timestamps, physics,
stroke options, recurrent phase labels, or target actions from held drops.
"""
import argparse
from collections import deque
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .tt24_bounce import BounceArena
from .tt24_motor import optical_sightings


class Perception:
    def __init__(self, obs):
        self.home = obs['joints'].copy()
        self.issued = np.zeros(5)
        self.track = deque(maxlen=8)
        self.last = np.zeros(3)
        self.age = 8

    def read(self, obs, blind=False):
        ball, _ = (None,None) if blind else optical_sightings(obs['rgb'])
        if ball is None:
            self.age = min(8,self.age+1)
        else:
            self.last = np.r_[ball.center, np.log(ball.radius / .08)]
            self.age = 0
        self.track.append(self.last)
        t = np.array(self.track)
        short = (t[-1]-t[max(0,len(t)-3)]) / min(2,max(len(t)-1,1))
        long = (t[-1]-t[0]) / max(len(t)-1,1)
        return np.r_[self.last, short, long, self.age,
                     obs['joints']-self.home, obs['joint_velocity'], self.issued, self.home]


class LocalFeedback:
    def __init__(self, rows):
        x, y = rows
        self.x, self.y = x, y
        self.scale = np.r_[[.25,.4,.9], [.10,.14,.32], [.07,.08,.20], 4.,
                           np.full(5,.25), np.full(5,2.), np.full(5,.25), np.full(5,.35)]
        self.z = x/self.scale
        self.tree = cKDTree(self.z)

    def command(self, x, issued, dt):
        z = x/self.scale
        dist, ids = self.tree.query(z, k=min(len(self.z),30))
        weight = np.exp(-((dist-dist[0])/max(.3,.7*dist[-1]))**2)
        design = np.column_stack((np.ones(len(ids)), self.z[ids]-z))
        w = design.T * weight
        reg = np.eye(design.shape[1])*.35
        reg[0,0] = 0.
        offset, rate = np.split(np.linalg.solve(w @ design + reg, w @ self.y[ids])[0],2)
        # A bounded corrective action can be applied on the next frame.
        action = .83*(offset-issued)/(5.2*dt) + .17*rate
        return np.clip(action,-1.,1.)


def rollout(arena, seed, actor=None, *, teacher=False, query=False, split='held', regime='standard', blind=False):
    obs = arena.reset(seed=int(seed),split=split,regime=regime,render=not blind)
    sensing = Perception(obs)
    if teacher or query:
        from .tt24_bounce_ceiling import PrivilegedCeiling
        tutor = PrivilegedCeiling(arena)
    states, labels = [], []
    frame = 0
    done = False
    while not done:
        x = sensing.read(obs,blind)
        expert = None
        if teacher or (query and frame%2==0):
            expert = tutor.command()
            states.append(x)
            labels.append(np.r_[sensing.issued+expert*(5.2*arena.dt),expert])
        if teacher:
            command = expert
        else:
            command = np.zeros(5) if actor is None else actor.command(x,sensing.issued,arena.dt)
        obs, _, done = arena.step(command,render=not blind)
        sensing.issued += command*(5.2*arena.dt)
        frame += 1
    metrics = arena.judge_metrics()
    return dict(n=metrics['rebounds'], run=metrics['longest_consecutive_rebounds'], frames=frame), (states,labels)


def combine(groups):
    return tuple(np.concatenate(packets) for packets in zip(*groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--demos',type=int,default=4)
    parser.add_argument('--corrections',type=int,default=8)
    parser.add_argument('--held',type=int,default=12)
    parser.add_argument('--seed',type=int,default=1164611)
    parser.add_argument('--arm',default='stock')
    parser.add_argument('--regime',default='standard')
    args = parser.parse_args()
    env = BounceArena(view='wrist',width=96,height=72,arm=args.arm)
    rng=np.random.default_rng(args.seed)
    demos, practice, held = [rng.integers(1,2**31-1,n).tolist() for n in [args.demos,args.corrections,args.held]]
    groups=[]
    exposure=[]
    outcomes={}
    try:
        for seed in demos:
            r, data = rollout(env,seed,teacher=True,split='practice',regime=args.regime)
            groups.append(data)
            exposure.append(dict(seed=seed,kind='demonstration',labels=len(data[0]),**r))
        for i in range(len(practice)+1):
            actor=LocalFeedback(combine(groups))
            if i in (0,len(practice)):
                label='teacher_only' if i==0 else 'after_corrections'
                outcomes[label]=[rollout(env,s,actor,regime=args.regime)[0] for s in held]
                print(label,[r['run'] for r in outcomes[label]],flush=True)
            if i==len(practice):
                break
            r,data=rollout(env,practice[i],actor,query=True,split='practice',regime=args.regime)
            groups.append(data)
            exposure.append(dict(seed=practice[i],kind='pupil_with_teacher_labels',labels=len(data[0]),**r))
        outcomes['blind']=[rollout(env,s,actor,blind=True,regime=args.regime)[0] for s in held]
        outcomes['stationary']=[rollout(env,s,None,blind=True,regime=args.regime)[0] for s in held]
        x,y=combine(groups)
        np.savez_compressed(args.checkpoint,x=x,y=y)
        with gzip.open(args.output,'wt') as f:
            json.dump(dict(arm=args.arm, regime=args.regime, exposure=exposure,held=held,outcomes=outcomes),f)
    finally:
        env.close()


if __name__=='__main__':
    main()
