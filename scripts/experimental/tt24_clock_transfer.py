"""Infer the tempo of a new physical environment from observed motion."""
import argparse
from collections import deque
import gzip
import json
from pathlib import Path
import numpy as np
import mujoco

from .tt24_bounce import BounceArena
from .tt24_bounce_ident import WristPupil, _features, _INC

DIR=Path('artifacts/tt24')
CP=DIR/'bounce-ident-pupil.npz'


class MotionTrace:
    def __init__(self):
        self.values=deque(maxlen=9)
        t=np.arange(-8,1,dtype=float)/60
        self.weights=2*np.linalg.pinv(np.column_stack((np.ones(9),t,t*t)))[2]

    def update(self,point):
        self.values.append(point.copy())
        if len(self.values)==9:
            return self.weights@np.array(self.values)


def source(arena,rng):
    samples=[]
    costs=[]
    for seed in rng.integers(1,2**31-1,6).tolist():
        pupil=WristPupil.load(CP)
        obs=arena.reset(seed=seed,split='practice')
        home=obs['joints'].copy()
        trace=MotionTrace()
        tick=0
        while True:
            state=pupil.sensor.observe(obs)
            accel=trace.update(state[:3])
            if accel is not None:
                samples.append(accel)
            u=pupil.actor.command(_features(state,obs['joints'],home),pupil.issued)
            pupil.issued+=_INC*u
            obs,_,done=arena.step(u)
            tick+=1
            if done:break
        costs.append((seed,tick))
    # Ball-free periods and impacts are outliers for the dominant visual motion.
    return np.median(samples,axis=0),costs


def evaluate(arena,seeds,reference,gravity,mode):
    recent=deque(maxlen=160)
    rows=[]
    for seed in seeds:
        pos=MotionTrace()
        pupil=WristPupil.load(CP)
        obs=arena.reset(seed=seed,split='held')
        home=obs['joints'].copy()
        tape,scales=[],[]
        while True:
            state=pupil.sensor.observe(obs)
            accel=pos.update(state[:3])
            if accel is not None:
                recent.append(accel)
            tempo=1.
            if mode=='diagnostic':
                tempo=np.sqrt(gravity)
            elif mode=='online' and len(recent)>=24:
                measured=np.median(recent,axis=0)
                ratio=np.dot(measured,reference)/np.dot(reference,reference)
                tempo=np.sqrt(np.clip(ratio,.48,1.90))
            state[3:6]/=tempo
            u=pupil.actor.command(_features(state,obs['joints'],home),pupil.issued)
            u=np.clip(tempo*u,-1,1) if mode!='hold' else np.zeros(5)
            pupil.issued+=_INC*u
            tape.append(u.tolist())
            scales.append(float(tempo))
            obs,_,done=arena.step(u)
            if done:break
        judge=arena.judge_metrics()
        arena.reset(seed=seed,split='held',render=False)
        for u in tape:
            arena.step(u,render=False)
        if arena.judge_metrics()!=judge:
            raise ValueError('physical tape diverged')
        rows.append(dict(seed=seed,run=judge['longest_consecutive_rebounds'],score=judge['rebounds'],
                         scale=scales,ticks=len(tape),actions=tape))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--seed',type=int,default=45181092)
    p.add_argument('--count',type=int,default=8)
    args=p.parse_args()
    rng=np.random.default_rng(args.seed)
    env=BounceArena(view='wrist',width=192,height=144,max_steps=450)
    try:
        reference,cost=source(env,rng)
    finally:
        env.close()
    doc=dict(reference=reference.tolist(),nominal_experience=cost,verified=True,regimes=[])
    for gravity,mass in ((.69,.81),(1.38,1.34)):
        env=BounceArena(view='wrist',width=192,height=144,max_steps=450)
        model=env._model
        model.opt.gravity[:]*=gravity
        model.body_mass[env._ball_body]*=mass
        model.body_inertia[env._ball_body]*=mass
        mujoco.mj_setConst(model,env._data)
        seeds=rng.integers(1,2**31-1,args.count).tolist()
        reg=dict(gravity=gravity,mass=mass,seeds=seeds,modes={})
        doc['regimes'].append(reg)
        try:
            for mode in ('native','online','diagnostic','hold'):
                rows=evaluate(env,seeds,reference,gravity,mode)
                reg['modes'][mode]=rows
                print(gravity,mode,[r['run'] for r in rows],flush=True)
                with gzip.open(args.out,'wt') as f:
                    json.dump(doc,f,separators=(',',':'))
        finally:
            env.close()


if __name__=='__main__':
    main()
