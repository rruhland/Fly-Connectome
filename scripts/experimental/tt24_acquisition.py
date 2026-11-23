"""Learn transient full-joint feedback on difficult donor releases without target practice."""
import argparse
import ctypes
import gzip
import json
from pathlib import Path

import numpy as np

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_broad_donors import FamilyWorld, FamilyArena, TRAIN, NOVEL, fixed_seeds
from .tt24_donor_assess import CHECKPOINT, ShiftedBounce
from .tt24_donor_trial import ROOT, FUSED_OUTPUT, Native, _live, load_public
from .tt24_launch_context import draw
from .tt24_posture import WidePosture


def bind(native):
    func=native.so.tt24_donor_transient
    p,i=ctypes.c_void_p,ctypes.c_int
    func.argtypes=[p,p,p,i,p,p,i,i,p]
    func.restype=None
    return func


def assess(func,world,w,candidates,batch,cap=180):
    x,physics=map(np.ascontiguousarray,batch)
    candidates=np.ascontiguousarray(candidates).reshape(-1,5,20)
    result=np.empty((len(candidates),len(x),4))
    func(world.handle,w.ctypes.data,candidates.ctypes.data,len(candidates),x.ctypes.data,
         physics.ctypes.data,len(x),cap,result.ctypes.data)
    return result


def budget(rows):
    scores=rows[...,1]
    merits=(.45*np.minimum(scores,5)/5+.25*(scores>=2)+.20*(scores>=5)+.10*np.minimum(scores,10)/10)
    by_body=merits.mean(axis=-1)
    return .70*by_body.mean(axis=0)+.30*by_body.min(axis=0)


def train(native,w,out):
    rng=np.random.default_rng(814927003)
    with gzip.open(ROOT/"tt24_donor-refined.json.gz","rt") as f:
        effects=np.array(json.load(f)["exploration_basis"])
    worlds=[FamilyWorld(native,TRAIN[i]) for i in (0,1,2,3,4)]
    func=bind(native)
    base=np.zeros_like(w)
    pools,weak,dev,log=[],[],[],[]
    try:
        for i,world in enumerate(worlds):
            batch=world.prepare(fixed_seeds(100,1980261+i),"held")
            pools.append(batch)
            baseline=assess(func,world,w,base,batch)[0,:,1]
            low=np.argsort(baseline,kind="stable")[:45]
            weak.append(low)
            dev.append(world.prepare(fixed_seeds(16,6801157+i),"held"))
        def test(options):
            return np.array([assess(func,world,w,options,batch) for world,batch in zip(worlds,dev)])
        champ=budget(test(base))[0]
        first=float(champ)
        best=base.copy()
        for gen in range(44):
            directions=[]
            for n in range(8):
                motor=effects[:,rng.integers(5)] if n%2==0 else rng.normal(size=5)
                sense=rng.normal(size=20)
                sense[rng.random(20)<.58]=0.
                direction=np.outer(motor,sense)+(.08 if n<6 else .35)*rng.normal(size=w.shape)
                norm=np.linalg.norm(direction)
                directions.append(direction*rng.uniform(.13,.55)/norm)
            directions=np.array(directions)
            options=np.concatenate([best[None],best+directions,best-directions])
            trials=[]
            for i in rng.choice(len(worlds),3,replace=False):
                indices=np.r_[rng.choice(weak[i],2,replace=False),rng.choice(100,2,replace=False)]
                batch=[a[indices] for a in pools[i]]
                trials.append(assess(func,worlds[i],w,options,batch))
            utility=budget(np.stack(trials))
            winner=1+int(utility[1:].argmax())
            checked=test(options[winner:winner+1])
            score=float(budget(checked)[0])
            if score>champ:
                best,champ=options[winner].copy(),score
            log.append(dict(generation=gen+1,training_max=float(utility[winner]),
                            dev=score,incumbent=champ,dev_by_body=checked[:,0,:,1].mean(axis=1).tolist()))
        np.savez_compressed(out/"acquisition-source.npz",anchor=w,transient=best)
        with gzip.open(out/"acquisition-training.json.gz","wt") as f:
            json.dump(dict(donor_baseline=first,donor_selected=champ,pool_drops=500,training_drops=44*17*12,
                           validation_drops=45*80,development_history=log),f)
    finally:
        for world in worlds:
            world.close()


def rollout(arena,native,seed,regime,anchor,residual,mode):
    obs=arena.reset(seed=seed,split="held",regime=regime)
    observer=load_public(FUSED_OUTPUT)
    issued=np.zeros(5)
    tape=[]
    for tick in range(arena.max_steps):
        gain=np.exp(-tick/16) if tick<120 else 0.
        if mode=="frozen":
            gain=0.
        if mode=="permanent":
            gain=1.
        weights=np.ascontiguousarray(anchor+gain*residual)
        action=draw(native,observer.observe(obs),weights,issued)
        tape.append(action.tolist())
        obs,_,done=arena.step(action)
        if done:
            break
    got=arena.judge_metrics()
    replay=_live(native,arena,anchor,seed,regime,"replay",tape=tape)
    if digest_json(got)!=digest_json(replay["metrics"]):
        raise ValueError("physical action replay mismatch")
    return dict(seed=seed,run=got["longest_consecutive_rebounds"],ticks=len(tape),
                rebounds=got["rebounds"],actions=tape)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",action="store_true")
    p.add_argument("--out",type=Path,default=ROOT)
    p.add_argument("--test",choices=["stock","tilted","novel","wide"])
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    native=Native()
    try:
        if args.train:
            with np.load(CHECKPOINT,allow_pickle=False) as d:
                train(native,np.ascontiguousarray(d["weights"]),args.out)
        if args.test:
            idx=("stock","tilted","novel","wide").index(args.test)
            with np.load(args.out/"acquisition-source.npz",allow_pickle=False) as d:
                w,extra=d["anchor"],d["transient"]
            arena=(lambda: BounceArena(view="wrist",width=192,height=144,max_steps=480),ShiftedBounce,
                   lambda: FamilyArena(NOVEL,view="wrist"),
                   lambda: WidePosture(view="wrist",width=192,height=144,max_steps=300))[idx]()
            try:
                seeds=fixed_seeds(24,23059410+idx)
                regime="standard" if idx in (0,3) else "changed"
                groups={}
                for mode in ("transient","frozen","permanent"):
                    groups[mode]=[rollout(arena,native,s,regime,w,extra,mode) for s in seeds]
                    print(args.test,mode,sum(a["run"] for a in groups[mode]),flush=True)
                with gzip.open(args.out/f"acquisition-{args.test}.json.gz","wt") as f:
                    json.dump(groups,f,separators=(",",":"))
            finally:
                arena.close()
    finally:
        native.working.cleanup()


if __name__=="__main__":
    main()
