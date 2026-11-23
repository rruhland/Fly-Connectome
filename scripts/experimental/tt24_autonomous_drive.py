"""Learn and recognize a changed motor interface after surprise, without a switch announcement."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_carried_wrist import CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_effects import transmission
from .tt24_forward_wiring import BallFree
from .tt24_motor_settle import I, STEP, inspect


from .tt24_motor_bank import decoder


class MotorMemory:
    def __init__(self):
        self.maps=[I.copy()]
        self.current=I.copy()
        self.inverse=I.copy()
        self.pending=False
        self.checks=0

    def start(self,joints):
        self.target=joints.astype(float).copy()
        self.pred=self.target.copy()
        self.evidence=0.

    def observe(self,command,joints):
        self.target+=STEP*(self.current@command)
        self.pred+=.25*(self.target-self.pred)
        discrepancy=np.sqrt(np.mean((joints-self.pred)**2))
        self.evidence=max(0.,self.evidence+(1. if discrepancy>.068 else -.50))
        if self.evidence>=4.:
            self.pending=True
        return float(discrepancy)

    def remember(self,result,info):
        self.current=result
        self.inverse=decoder(result)
        if info["known"] is None:
            self.maps.append(result)
        self.pending=False
        self.checks+=1


def course(d,t):
    if d==1:
        return int(t>=70)
    if 2<=d<=4:
        return 1
    if 5<=d<=7:
        return 2
    if d==8:
        return 2 if t<70 else 0
    if 11<=d<=13:
        return 1
    return 0


def body_for(d):
    return "alternate" if 5<=d<=8 else "stock"


def play(env,policy,memory,seed,d,maps,mode,tape=None):
    body=body_for(d)
    regime="changed" if body=="alternate" else "standard"
    obs=env.reset(seed=seed,split="held",regime=regime,render=tape is None)
    if tape is None:
        policy.begin_release(obs)
        memory.start(obs["joints"])
    actions=[]
    flagged=None
    while True:
        t=len(actions)
        k=course(d,t)
        if tape is None:
            desired=policy.act(obs)
            if mode=="known":
                transform=decoder(maps[k])
            elif mode=="adaptive":
                transform=memory.inverse
            else:
                transform=I
            command=np.clip(transform@desired,-1.,1.)
            applied=maps[k]@command
        else:
            applied=tape[t]
        obs,_,done=env.step(applied,render=tape is None)
        actions.append(np.asarray(applied).tolist())
        if tape is None and mode=="adaptive":
            memory.observe(command,obs["joints"])
            if memory.pending and flagged is None:
                flagged=t
        if done:
            break
    return dict(actions=actions,metrics=env.judge_metrics(),flagged=flagged)


def run(index):
    rng=np.random.default_rng(52480514+index)
    maps=[I,transmission(rng),transmission(rng)@(I+rng.uniform(-.14,.14,(5,5)))]
    plans=rng.integers(1,2**31-1,(16,3)).tolist()
    bodies=("stock","alternate")
    arenas={body:BounceArena(arm=body,view="wrist",width=192,height=144,max_steps=420) for body in bodies}
    benches={body:BallFree(body,100) for body in bodies}
    policy=CarriedWristPolicy()
    report={}
    try:
        for mode in ("adaptive","frozen","known"):
            memory=MotorMemory()
            result=[]
            for d,(seed,probe_seed,directions) in enumerate(plans):
                body=body_for(d)
                env=arenas[body]
                inspection=None
                if mode=="adaptive" and memory.pending:
                    fit,inspection=inspect(benches[body],probe_seed,maps[course(d,0)],memory.maps,directions,
                                           "changed" if body=="alternate" else "standard")
                    memory.remember(fit,inspection)
                row=play(env,policy,memory,seed,d,maps,mode)
                replay=play(env,policy,memory,seed,d,maps,mode,row["actions"])
                if replay["actions"]!=row["actions"] or replay["metrics"]!=row["metrics"]:
                    raise ValueError("physical replay mismatch")
                result.append(row|dict(body=body,inspection=inspection,index=policy.calibration_index,
                                       map_error=float(np.linalg.norm(memory.current-maps[course(d,0)]))))
            report[mode]=result
        with gzip.open(ROOT/f"autonomous-drive-{index}.json.gz","wt") as f:
            json.dump(dict(plans=plans,wires=[m.tolist() for m in maps],modes=report),f,separators=(",",":"))
    finally:
        for arena in (*arenas.values(),*benches.values()):
            arena.close()


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(run,range(4)))
