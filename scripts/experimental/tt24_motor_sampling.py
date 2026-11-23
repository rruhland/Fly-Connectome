"""Bounded small motor probes stop once measured motion has settled, then test gameplay."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_carried_wrist import CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_effects import transmission
from .tt24_forward_wiring import BallFree
from .tt24_motor_settle import STEP, I, game, inspect


from .tt24_motor_bank import sample


def runner(index):
    body="alternate" if index%2 else "stock"
    regime="changed" if body=="alternate" else "standard"
    rng=np.random.default_rng(551805412+index)
    bench=BallFree(body,260)
    env=BounceArena(arm=body,view="wrist",width=192,height=144,max_steps=420)
    policy=CarriedWristPolicy()
    cases=[]
    try:
        for typ in (0,1):
            w=transmission(rng)
            if typ:
                w=w@(I+rng.uniform(-.15,.15,(5,5)))
            probe_seed,signals=rng.integers(1,2**31-1,2).tolist()
            old,baseline=inspect(bench,probe_seed,w,[I],signals,regime)
            obs=bench.reset(seed=probe_seed,split="held",regime=regime,render=False)
            def step(u):
                o,_,_=bench.step(w@u,render=False)
                return o["joints"]
            learned,sampled=sample(obs["joints"],step,signals)
            games=[]
            for seed in rng.integers(1,2**31-1,4).tolist():
                result={}
                for k,estimate in (("adaptive_settle",learned),("fixed_settle",old),("frozen",I),("known",w)):
                    row=game(env,policy,seed,regime,w,estimate)
                    if row!=game(env,policy,seed,regime,w,estimate,tape=row["tape"]):
                        raise ValueError("replay mismatch")
                    result[k]=row
                games.append(dict(seed=seed,conditions=result))
            cases.append(dict(wire=w.tolist(),signals=signals,probe_seed=probe_seed,
                              adaptive=sampled,baseline=baseline|dict(fit=old.tolist()),games=games))
        with gzip.open(ROOT/f"motor-sampling-{index}.json.gz","wt") as f:
            json.dump(dict(body=body,cases=cases),f,separators=(",",":"))
    finally:
        bench.close()
        env.close()


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(runner,range(4)))
