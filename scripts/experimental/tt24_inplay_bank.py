"""Recognize previously measured motor maps from ordinary closed-loop movements inside a drop."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import gzip
import json

import numpy as np

from .tt24_autonomous_drive import MotorMemory, body_for, course
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_forward_wiring import BallFree
from .tt24_motor_bank import I, STEP, decoder
from .tt24_motor_settle import inspect


class Recalling(MotorMemory):
    def start(self,joints):
        super().start(joints)
        self.history=deque([(np.zeros(5),joints.astype(float))],maxlen=23)
        self.smoothed=np.zeros(5)
        self.winning=None
        self.consecutive=0
        self.ticks=0
        self.changed=[]
        self.flagged=None

    def notice(self,command,joints):
        self.ticks+=1
        self.smoothed+=.25*(command-self.smoothed)
        total=self.history[-1][0]+STEP*self.smoothed
        self.history.append((total,joints.astype(float)))
        super().observe(command,joints)
        if self.pending and self.flagged is None:
            self.flagged=self.ticks
        lengths=[w for w in (6,12,20) if len(self.history)>w]
        if not lengths or len(self.maps)<2:
            return False
        motion=np.array([total-self.history[-w-1][0] for w in lengths])
        measured=np.array([joints-self.history[-w-1][1] for w in lengths])
        if np.linalg.norm(motion[-1])<.05:
            return False
        rms=np.array([np.sqrt(np.mean((motion@m.T-measured)**2)) for m in self.maps])
        candidate=int(rms.argmin())
        current=next(i for i,m in enumerate(self.maps) if np.array_equal(m,self.current))
        if candidate==current:
            self.winning=None
            self.consecutive=0
            return False
        margin=(rms[candidate]+.009)*1.65 < rms[current]+.009
        if not (margin and rms[candidate]<.038):
            self.consecutive=0
            return False
        self.consecutive=self.consecutive+1 if self.winning==candidate else 1
        self.winning=candidate
        if self.consecutive<3:
            return False
        self.changed.append([self.ticks,candidate,rms.tolist()])
        self.current=self.maps[candidate]
        self.inverse=decoder(self.current)
        self.pred=joints.astype(float).copy()
        self.target=self.pred.copy()
        self.pending=False
        self.evidence=0.
        self.consecutive=0
        return True


def actual(env,policy,mem,seed,d,wires,tape=None):
    obs=env.reset(seed=seed,split="held",regime="changed" if body_for(d)=="alternate" else "standard",
                  render=tape is None)
    if tape is None:
        policy.begin_release(obs)
        reference=obs["joints"]-policy.issued
        mem.start(obs["joints"])
    commands=[]
    while True:
        t=len(commands)
        if tape is None:
            sent=np.clip(mem.inverse@policy.act(obs),-1.,1.)
            physical=wires[course(d,t)]@sent
        else:
            physical=tape[t]
        commands.append(np.asarray(physical).tolist())
        obs,_,done=env.step(physical,render=tape is None)
        if tape is None and mem.notice(sent,obs["joints"]):
            policy.issued=obs["joints"].astype(float)-reference
        if done:
            break
    return commands,env.judge_metrics()


def run(index):
    with gzip.open(ROOT/f"autonomous-drive-{index}.json.gz","rt") as f:
        ref=json.load(f)
    wires=list(map(np.asarray,ref["wires"]))
    bodies=("stock","alternate")
    envs={b:BounceArena(arm=b,view="wrist",width=192,height=144,max_steps=420) for b in bodies}
    benches={b:BallFree(b,100) for b in bodies}
    p=CarriedWristPolicy()
    mem=Recalling()
    rows=[]
    try:
        for d,(seed,probe,directions) in enumerate(ref["plans"]):
            b=body_for(d)
            log=None
            if mem.pending:
                m,log=inspect(benches[b],probe,wires[course(d,0)],mem.maps,directions,
                              "changed" if b=="alternate" else "standard")
                mem.remember(m,log)
            a,r=actual(envs[b],p,mem,seed,d,wires)
            replay,re=actual(envs[b],p,mem,seed,d,wires,a)
            if re!=r or replay!=a:
                raise ValueError("in-play map replay failed")
            rows.append(dict(seed=seed,actions=a,metrics=r,exchanges=mem.changed,inspection=log,alarm=mem.flagged))
        with gzip.open(ROOT/f"inplay-bank-{index}.json.gz","wt") as f:
            json.dump(dict(rows=rows),f,separators=(",",":"))
    finally:
        for env in (*envs.values(),*benches.values()):
            env.close()


if __name__=="__main__":
    import sys
    jobs=list(map(int,sys.argv[1:])) or list(range(4))
    with ProcessPoolExecutor(max_workers=min(2,len(jobs))) as pool:
        list(pool.map(run,jobs))
