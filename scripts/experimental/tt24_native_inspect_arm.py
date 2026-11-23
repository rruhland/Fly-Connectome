"""Carry an inspection threshold fitted on native Pong outcomes through changing five-axis wrist play."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json

import numpy as np

from .tt24_autonomous_drive import body_for, course
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_effects import transmission
from .tt24_forward_wiring import BallFree
from .tt24_motor_bank import I, STEP, decoder
from .tt24_motor_settle import inspect


def play(env,p,seed,d,transform,wires,tape=None):
    regime="standard" if body_for(d)=="stock" else "changed"
    obs=env.reset(seed=seed,split="held",regime=regime,render=tape is None)
    q=[obs["joints"].astype(float).tolist()]
    if tape is None:
        p.begin_release(obs)
    commands=[]
    actions=[]
    while True:
        tick=len(actions)
        if tape is None:
            nominal=p.act(obs)
            sent=np.clip(transform@nominal,-1.,1.)
            applied=wires[course(d,tick)]@sent
            commands.append(sent.tolist())
        else:
            applied=tape[tick]
        obs,_,done=env.step(applied,render=tape is None)
        q.append(obs["joints"].astype(float).tolist())
        actions.append(np.asarray(applied).tolist())
        if done:
            break
    return dict(metrics=env.judge_metrics(),actions=actions,q=q,commands=commands)


def alarm(row,m,threshold):
    q=np.array(row["q"])
    target=pred=q[0]
    evidence=0.
    for t,(cmd,measured) in enumerate(zip(row["commands"],q[1:],strict=True)):
        target=target+STEP*(m@cmd)
        pred=pred+.25*(target-pred)
        error=np.sqrt(np.mean((pred-measured)**2))/STEP
        evidence=max(0.,evidence+(1. if error>threshold else -.5))
        if evidence>=4.:
            return t
    return None


def run(index):
    with gzip.open(ROOT/"pong-inspection-held.json.gz","rt") as f:
        source=json.load(f)["streams"][index]["chosen"]
    thresholds={k:source[k]["threshold"] for k in ("pong","shuffled")}
    thresholds.update(arm=.068/STEP,high=3.3,never=None)
    rng=np.random.default_rng(161863151+index)
    wires=[I,transmission(rng),transmission(rng)@(I+rng.uniform(-.14,.14,(5,5)))]
    plans=rng.integers(1,2**31-1,(16,3)).tolist()
    bodies=("stock","alternate")
    envs={b:BounceArena(arm=b,view="wrist",width=192,height=144,max_steps=360) for b in bodies}
    benches={b:BallFree(b,100) for b in bodies}
    p=CarriedWristPolicy()
    cache={}
    bank_cache={}
    training={}
    try:
        for name,threshold in thresholds.items():
            memory=[I.copy()]
            m=I.copy()
            signalled=False
            rows=[]
            for d,(s,probe,draw) in enumerate(plans):
                b=body_for(d)
                setup=None
                if signalled:
                    key=(d,tuple(z.tobytes() for z in memory))
                    if key not in bank_cache:
                        bank_cache[key]=inspect(benches[b],probe,wires[course(d,0)],memory,draw,
                                                     "changed" if b=="alternate" else "standard")
                    m,setup=bank_cache[key]
                    if setup["known"] is None:
                        memory.append(m)
                key=(d,m.tobytes())
                if key not in cache:
                    row=play(envs[b],p,s,d,decoder(m),wires)
                    replay=play(envs[b],p,s,d,decoder(m),wires,row["actions"])
                    if row["metrics"]!=replay["metrics"] or row["q"]!=replay["q"]:
                        raise ValueError("physical trace mismatch")
                    cache[key]=row|dict(trace=len(cache))
                result=cache[key]
                first=None if threshold is None else alarm(result,m,threshold)
                signalled=first is not None
                rows.append(dict(drop=d,trace=result["trace"],alarm=first,setup=setup,
                                 rebounds=result["metrics"]["rebounds"],run=result["metrics"]["longest_consecutive_rebounds"]))
            training[name]=rows
        traces=sorted(cache.values(),key=lambda r:r["trace"])
        with gzip.open(ROOT/f"native-inspect-arm-{index}.json.gz","wt") as f:
            json.dump(dict(index=index,thresholds=thresholds,plans=plans,modes=training,traces=traces,
                           wires=[m.tolist() for m in wires]),f,separators=(",",":"))
    finally:
        for e in (*envs.values(),*benches.values()):
            e.close()


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(run,range(4)))
