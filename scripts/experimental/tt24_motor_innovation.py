"""Recover newly effective controls from learned encoder transitions during normal wrist play."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import gzip
import json

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_carried_drive import simulate
from .tt24_carried_wrist import CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_effects import transmission
from .tt24_motor_recall import sensor
from .tt24_predict_witness import BodyPlay


def design(states):
    # Position and recent velocity differences capture local servo response regardless of body task.
    q = states[2:-1,:5]
    v = states[2:-1,5:]
    past = states[1:-2,5:]
    older = states[:-3,5:]
    new = states[3:,5:]
    terms = [q, v, new-v, v-past, past-older]
    mechanical = np.c_[v,new-v,v-past]
    terms.append((q[:,:,None] * mechanical[:,None,:]).reshape(len(v),-1))
    return np.column_stack(terms)


def probes():
    arena=BodyPlay()
    rows=[]
    try:
        for i in range(32):
            seed=3515811+i
            rng=np.random.default_rng(seed)
            obs=arena.reset(seed=seed,split="held",render=False)
            origin=obs["joints"].copy()
            states=[sensor(obs)]
            drives=[]
            signal=np.zeros(5)
            for t in range(100):
                scale=(.09,.23,.42,.17)[t//25]
                signal=.25*signal+rng.normal(0.,scale,5)
                u=np.clip(signal-1.6*(obs["joints"]-origin),-.65,.65)
                obs,_,_=arena.step(u,render=False)
                states.append(sensor(obs))
                drives.append(u)
            rows.append((design(np.asarray(states)),np.asarray(drives)[2:]))
    finally:
        arena.close()
    return rows


class Innovation:
    def __init__(self, params):
        self.mean,self.std,self.weights = (params[k] for k in ("mean","std","weights"))
        self.rng=np.random.default_rng(108399)
        self.eye=np.eye(5)
        self.known=[self.eye.copy()]
        self.index=0
        self.detect=0
        self.collecting=False
        self.C=self.eye.copy()
        self.S=np.zeros((5,5))
        self.Y=np.zeros((5,5))
        self.count=0

    def reset(self,obs):
        self.buffer=deque([sensor(obs)], maxlen=4)
        self.events=[]
        self.ticks=0

    def actuate(self,desired):
        if self.collecting:
            # Independent small commands make unknown motor directions observable while playing.
            return np.clip(self.C@desired + self.rng.uniform(-.19,.19,5),-1.,1.)
        return np.clip(self.C@desired,-1.,1.)

    def observe(self,u,obs,pupil):
        self.buffer.append(sensor(obs))
        self.ticks+=1
        if len(self.buffer)<4:
            return
        x=design(np.array(self.buffer))
        w=(x[0]-self.mean)/self.std
        effect=self.weights[0]+w@self.weights[1:]
        error=np.linalg.norm(effect-self.known[self.index]@u)
        if not self.collecting:
            self.detect=max(0,self.detect+(1 if error>.16+np.linalg.norm(u)*.35 else -1))
            if self.detect<3:
                return
            self.events.append([self.ticks,"detect"])
            self.collecting=True
            self.S=self.eye*.06
            self.Y=self.eye*.06
            self.count=0
        self.S+=np.outer(u,u)
        self.Y+=np.outer(effect,u)
        self.count+=1
        if self.count<16 or np.linalg.eigvalsh(self.S)[0]<.19:
            return
        fit=np.linalg.solve(self.S,self.Y.T).T
        distances=[float(np.linalg.norm(fit-m)) for m in self.known]
        nearby=int(np.argmin(distances))
        if distances[nearby]<.93:
            chosen=self.known[nearby]
            self.index=nearby
        else:
            chosen=fit
            self.known.append(chosen)
            self.index=len(self.known)-1
        if self.index==0:
            self.C=self.eye.copy()
        else:
            left,s,right=np.linalg.svd(chosen)
            self.C=(right.T*(s/(s*s+.018)))@left.T
        self.events.append([self.ticks,"map",self.index,self.count,distances[nearby]])
        self.detect=0
        self.collecting=False


def fit_model():
    rows=probes()
    x,y=map(np.concatenate,zip(*rows[:24],strict=True))
    mean=x.mean(axis=0)
    std=np.maximum(x.std(axis=0),1.e-3)
    x=np.c_[np.ones(len(x)),(x-mean)/std]
    weights=np.linalg.solve(x.T@x+.9*np.eye(x.shape[1]),x.T@y)
    test,target=map(np.concatenate,zip(*rows[24:],strict=True))
    pred=np.c_[np.ones(len(test)),(test-mean)/std]@weights
    score=np.mean((pred-target)**2,axis=0)**.5
    np.savez_compressed(ROOT/"motor-innovation-observer.npz",mean=mean,std=std,weights=weights,
                        held_action_rms=score,target_rms=(target**2).mean(axis=0)**.5)
    print("held inverse RMS",score,flush=True)


def run(index):
    rng=np.random.default_rng(77864326+index)
    seeds=rng.integers(1,2**31-1,9).tolist()
    wiring=transmission(rng)
    env=BounceArena(view="wrist",width=192,height=144,max_steps=480)
    p=CarriedWristPolicy()
    with np.load(ROOT/"motor-innovation-observer.npz",allow_pickle=False) as params:
        adapt=Innovation(params)
    rows=[]
    try:
        for d,seed in enumerate(seeds):
            row=simulate(env,p,adapt,wiring,seed,d,"adaptive")
            replay=simulate(env,p,adapt,wiring,seed,d,"adaptive",row["tape"])
            if row["metrics"]!=replay["metrics"] or row["events"]!=replay["events"]:
                raise ValueError("physical replay failed")
            rows.append(row)
    finally:
        env.close()
    with gzip.open(ROOT/f"motor-innovation-{index}.json.gz","wt") as f:
        json.dump(dict(modes=len(adapt.known),rows=rows),f,separators=(",",":"))


if __name__=="__main__":
    fit_model()
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(run,range(4)))
