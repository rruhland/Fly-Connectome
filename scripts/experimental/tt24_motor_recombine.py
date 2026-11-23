"""First-use transfer of a library of previously learned interfaces between different physical arms."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_carried_wrist import CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_effects import transmission
from .tt24_forward_wiring import BallFree
from .tt24_motor_bank import MotorBank
from .tt24_motor_settle import I, game


def run(body, measured=False):
    regime="changed" if body=="alternate" else "standard"
    previous=0 if body=="alternate" else 1
    rng=np.random.default_rng((885361431 if measured else 351456716)+previous)
    prefix="motor-recombine-motion" if measured else "motor-recombine"
    if measured:
        with gzip.open(ROOT/f"motor-sampling-{previous}.json.gz","rt") as f:
            other=json.load(f)
        wires=[I]+[np.array(c["wire"]) for c in other["cases"]]
        bank=MotorBank.load(ROOT/"motor-bank-motion.npz")
    else:
        with gzip.open(ROOT/f"motor-settle-{previous}.json.gz","rt") as f:
            other=json.load(f)
        wires=[np.array(w) for w in other["wires"]]
        bank=MotorBank.load(ROOT/"motor-bank.npz")
    wires.append(transmission(rng))
    cold=MotorBank()
    policy=CarriedWristPolicy()
    arena=BounceArena(arm=body,view="wrist",width=192,height=144,max_steps=360)
    bench=BallFree(body,260 if measured else 100)
    rows=[]
    try:
        # Source fits of both familiar nonidentity maps were obtained on the opposite body.
        for i in (1,2,3,0,1,3):
            s,r=rng.integers(1,2**31-1,2).tolist()
            logs={}
            maps={}
            for label,memory in (("carried",bank),("new_bank",cold)):
                obs=bench.reset(seed=s,split="held",regime=regime,render=False)
                def step(u):
                    snap,_,_=bench.step(wires[i]@u,render=False)
                    return snap["joints"]
                logs[label]=memory.connect(obs["joints"],step,r,measured_settling=measured)
                maps[label]=memory.maps[memory.selected]
                logs[label]["selected"]=memory.selected
            maps["true"]=wires[i]
            maps["unchanged"]=I
            games=[]
            for seed in rng.integers(1,2**31-1,3).tolist():
                results={}
                for key,m in maps.items():
                    prior=next((k for k in results if np.array_equal(m,maps[k])),None)
                    if prior is not None:
                        results[key]=results[prior]
                        continue
                    row=game(arena,policy,seed,regime,wires[i],m)
                    if row!=game(arena,policy,seed,regime,wires[i],m,tape=row["tape"]):
                        raise ValueError("recombination tape did not replay")
                    results[key]=row
                games.append(dict(seed=seed,results=results))
            rows.append(dict(interface=i,seed=s,directions=r,inspection=logs,games=games))
        bank.save(ROOT/f"{prefix}-{body}-bank.npz" if measured else ROOT/f"motor-bank-{body}-revisit.npz")
        # Learned memory persists exactly through a disk roundtrip.
        restored=MotorBank.load(ROOT/f"{prefix}-{body}-bank.npz" if measured else ROOT/f"motor-bank-{body}-revisit.npz")
        if not all(np.array_equal(a,b) for a,b in zip(restored.maps,bank.maps,strict=True)):
            raise ValueError("inexact serialized motor memory")
        with gzip.open(ROOT/f"{prefix}-{body}.json.gz","wt") as f:
            json.dump(dict(body=body,rows=rows,wires=[w.tolist() for w in wires]),f,separators=(",",":"))
    finally:
        bench.close()
        arena.close()


if __name__=="__main__":
    import sys
    from functools import partial
    use_measured="--measured" in sys.argv[1:]
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(partial(run,measured=use_measured),("stock","alternate")))
