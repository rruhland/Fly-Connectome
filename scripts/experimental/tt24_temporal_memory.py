"""Adapt native Pong's prediction to actual target motion using only subsequently delivered packets."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_donor_trial import ROOT, Native
from .tt24_posture import WidePosture
from .tt24_pong_temporal_wrist import DelayedWrist, MAX_STEPS, starts
from .tt24_ready import episode


class Memory:
    def __init__(self):
        self.capacity=768
        self.x=np.zeros((9,self.capacity,8))
        self.y=np.zeros((9,self.capacity,6))
        self.count=np.zeros(9,int)

    def save(self,path):
        np.savez_compressed(path,schema=np.array("tt24-temporal-memory-1"),x=self.x,y=self.y,count=self.count)

    @classmethod
    def load(cls,path):
        memory=cls()
        with np.load(path,allow_pickle=False) as checkpoint:
            if checkpoint["schema"].item()!="tt24-temporal-memory-1":
                raise ValueError("unrecognized forecast memory")
            for key in ("x","y","count"):
                value=checkpoint[key]
                if value.shape!=getattr(memory,key).shape or not np.isfinite(value).all():
                    raise ValueError("invalid forecast memory")
                setattr(memory,key,value)
        return memory

    @staticmethod
    def context(state):
        return np.r_[(state[:3]-state[6:9]) / (.065,.065,.13),
                     state[3:6] / (.38,.38,1.5),state[9:11]/.18]

    def store(self,age,state,future,gain):
        baseline=state[:6].copy()
        baseline[:3]+=gain*state[3:6]*age/60
        n=self.count[age]%self.capacity
        self.x[age,n]=self.context(state)
        self.y[age,n]=np.clip(future[:6]-baseline,(-.12,-.12,-.30,-1.,-1.,-3.),(.12,.12,.30,1.,1.,3.))
        self.count[age]+=1

    def correction(self,age,state):
        n=min(self.count[age],self.capacity)
        if n<8:
            return 0.
        d=np.sum((self.x[age,:n]-self.context(state))**2,axis=1)
        neighbours=np.argpartition(d,min(8,n)-1)[:8]
        weight=np.exp(-d[neighbours]/.70)
        return (weight@self.y[age,neighbours])/(1.5+weight.sum())


class LearnedForecast(DelayedWrist):
    def __init__(self,seed,gain,archive,*,update,delay=True):
        super().__init__(seed,gain,delay)
        self.archive=archive
        self.update=update
        self.recent={}

    def observe(self,live):
        old=self.capture
        result=super().observe(live)
        if self.capture>old and self.update:
            for t,prior in list(self.recent.items()):
                h=self.capture-t
                if h>8:
                    del self.recent[t]
                else:
                    self.archive.store(h,prior,self.last_state,self.gain)
            self.recent[self.capture]=self.last_state.copy()
        age=self.tick-self.capture
        if age:
            result[:6]+=self.archive.correction(age,self.last_state)
        return result


def run(index,verify_archive=False):
    rng=np.random.default_rng(67105512+index)
    training=starts(int(rng.integers(2**31)),12)
    held=starts(int(rng.integers(2**31)),6)
    with np.load(BUNDLE,allow_pickle=False) as d:
        w,mu=d["weights"],d["equilibrium"]
    with gzip.open(ROOT/"pong-temporal-source.json.gz","rt") as f:
        gain=json.load(f)["gains"]["pong"]
    n=Native()
    arena=WidePosture(view="wrist",width=192,height=144,max_steps=MAX_STEPS)
    launcher=BounceArena(view="none")
    recordings={}
    receipt=ROOT/f"temporal-memory-{index}.json.gz"
    if verify_archive:
        with gzip.open(receipt,"rt") as f:
            archived=json.load(f)["modes"]["online"]
    try:
        for kind in (("online",) if verify_archive else ("online","within","frozen","prompt")):
            state=Memory()
            series=[]
            for i,s in enumerate(training+held):
                if kind=="online" and i==len(training):
                    checkpoint=ROOT/f"temporal-memory-{index}.npz"
                    state.save(checkpoint)
                    state=Memory.load(checkpoint)
                if kind=="within":
                    state=Memory()
                if kind=="prompt":
                    factory=None
                elif kind=="frozen":
                    factory=lambda _,s=s: DelayedWrist(s,gain,True)
                else:
                    factory=lambda _,s=s: LearnedForecast(s,gain,state,update=i<len(training) or kind=="within")
                row=episode(arena,launcher,s,n,w,mu,"scalar","standard",alpha=1.,sensor_factory=factory)
                row["trained_pairs"]=state.count.tolist()
                if verify_archive and row!=archived[i]:
                    raise ValueError("loaded forecast model failed to reproduce its complete issued tape")
                series.append(row)
            recordings[kind]=series
        if verify_archive:
            print("verified", index, sum(len(r["actions"]) for r in series[len(training):]),flush=True)
            return
        with gzip.open(receipt,"wt") as f:
            json.dump(dict(gain=gain,train_count=len(training),modes=recordings),f,separators=(",",":"))
    finally:
        arena.close()
        launcher.close()
        n.working.cleanup()


if __name__=="__main__":
    import sys
    from functools import partial
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(partial(run,verify_archive="--verify-archive" in sys.argv),range(3)))
