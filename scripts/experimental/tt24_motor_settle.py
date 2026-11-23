"""Identify a new actuator interface rapidly from the settled effects of generic voluntary movements."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_carried_wrist import CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_effects import transmission
from .tt24_forward_wiring import BallFree

from .tt24_motor_bank import I, STEP, decoder, identify


def inspect(arena,seed,drive,known,coords,regime):
    obs=arena.reset(seed=seed,split="held",regime=regime,render=False)
    def step(request):
        measured,_,_=arena.step(drive@request,render=False)
        return measured["joints"]
    return identify(obs["joints"],step,known,coords)


def game(env,policy,seed,regime,wiring,estimate,tape=None):
    obs=env.reset(seed=seed,split="held",regime=regime,render=tape is None)
    records=[]
    if tape is None:
        policy.begin_release(obs)
        inverse=decoder(estimate)
    while True:
        physical=(wiring@np.clip(inverse@policy.act(obs),-1.,1.) if tape is None else tape[len(records)])
        records.append(np.asarray(physical).tolist())
        obs,_,done=env.step(physical,render=tape is None)
        if done:
            break
    return dict(tape=records,metrics=env.judge_metrics())


def stream(index):
    body=("stock","alternate")[index%2]
    regime="standard" if body=="stock" else "changed"
    env=BounceArena(arm=body,view="wrist",width=192,height=144,max_steps=420)
    bench=BallFree(body,100)
    p=CarriedWristPolicy()
    rng=np.random.default_rng(451109324+index)
    wires=[I,transmission(rng),transmission(rng)]
    # Add a different cross-axis shear; the learner does not receive the family or draw.
    wires[2]=wires[2]@(I+rng.uniform(-.15,.15,(5,5)))
    archive=[I.copy()]
    sessions=[]
    try:
        for k in (0,1,2,0,1):
            seed=int(rng.integers(1,2**31-1))
            draw=int(rng.integers(1,2**31-1))
            fit,log=inspect(bench,seed,wires[k],archive,draw,regime)
            if log["known"] is None:
                archive.append(fit)
            log.update(map_error=float(np.linalg.norm(fit-wires[k])/np.linalg.norm(wires[k])),
                       fit=fit.tolist(),seed=seed,excitation_seed=draw)
            games=[]
            for s in rng.integers(1,2**31-1,4).tolist():
                result={}
                for name,m in (("calibrated",fit),("frozen",I),("known",wires[k])):
                    # Identical settings on the identical seed execute the same physical game.
                    existing=next((v for key,v in result.items() if np.array_equal(m,{"calibrated":fit,"frozen":I}[key])),None)
                    if existing is not None:
                        result[name]=existing
                        continue
                    row=game(env,p,s,regime,wires[k],m)
                    if row!=game(env,p,s,regime,wires[k],m,tape=row["tape"]):
                        raise ValueError("changed drive replay mismatch")
                    result[name]=row
                games.append(dict(seed=s,conditions=result))
            sessions.append(dict(interface=k,inspection=log,games=games))
        with gzip.open(ROOT/f"motor-settle-{index}.json.gz","wt") as f:
            json.dump(dict(body=body,sessions=sessions,wires=[w.tolist() for w in wires]),f,separators=(",",":"))
    finally:
        env.close()
        bench.close()


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(stream,range(4)))
