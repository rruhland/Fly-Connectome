"""Learn a reference during play as the arm moves between a fixed and a mobile serving station."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import gzip
import json

import numpy as np

from .tt24_bounce import BounceArena, _sample_drop
from .tt24_broad_donors import fixed_seeds
from .tt24_donor_assess import CHECKPOINT
from .tt24_donor_trial import Native, ROOT
from .tt24_posture import WidePosture
from .tt24_ready import episode

# These motor actions set a reference coefficient. The actual five-joint command depends on each wrist frame.
VALUES=np.array([0.,.5,1.])


class SideStation(BounceArena):
    span=(.52,.83)

    @classmethod
    def sample_drop(cls,seed,split,regime):
        drop=_sample_drop(seed,split,regime)
        rng=np.random.default_rng(int(seed)^9412773)
        pose=np.asarray(drop.pose).copy()
        pose[0]+=rng.choice([-1.,1.])*rng.uniform(*cls.span)
        return replace(drop,pose=tuple(pose))


class ModerateStation(SideStation):
    span=(.18,.34)


def read_parameters():
    with np.load(CHECKPOINT,allow_pickle=False) as d:
        w=d["weights"]
    with np.load(ROOT/"posture-source.npz",allow_pickle=False) as d:
        center,cov=d["center"],d["covariance"]
    return w,center,cov


def course(kind,body,immediate=False):
    station=ModerateStation if immediate else SideStation
    player=WidePosture if kind==0 else station
    server=BounceArena if kind==0 else station
    return (player(view="wrist",width=192,height=144,max_steps=360,arm=body),
            server(view="none",arm=body))


def descriptor(first,center):
    # Raw encoder offsets and their absolute values. No station, position label or game score.
    q=(first["joints"]-center)/.22
    return np.r_[q,abs(q)]


def screening(immediate):
    native=Native()
    w,mu,_=read_parameters()
    groups={}
    try:
        for kind in (0,1):
            arena,server=course(kind,"stock",immediate)
            try:
                seeds=fixed_seeds(8,12193416+kind+100*int(immediate))
                groups[str(kind)]={}
                for alpha in VALUES:
                    rows=[episode(arena,server,s,native,w,mu,"scalar","standard",alpha=alpha,lead_override=0 if kind and immediate else None) for s in seeds]
                    groups[str(kind)][str(alpha)]=rows
                print(kind,[sum(r["run"] for r in groups[str(kind)][str(a)]) for a in VALUES],flush=True)
            finally:
                arena.close()
                server.close()
        with gzip.open(ROOT/("station-switch-immediate-feasibility.json.gz" if immediate else "station-switch-feasibility.json.gz"),"wt") as f:
            json.dump(groups,f,separators=(",",":"))
    finally:
        native.working.cleanup()


def stream(job):
    index,immediate=job
    native=Native()
    w,mu,_=read_parameters()
    body="stock" if index<4 else "alternate"
    regime="standard" if body=="stock" else "changed"
    arenas=[course(i,body,immediate) for i in (0,1)]
    stations=[0]*5+[1]*7+[0]*4+[1]*4
    seeds=fixed_seeds(len(stations),129572043+index)
    with gzip.open(ROOT/"tt24_pong_origin-bank.json.gz","rt") as f:
        origin=json.load(f)["replicas"][index]["selected"]
    modes={"native_memory":origin["native"],"shuffled_memory":origin["shuffled"],"cold_memory":.5,
           "native_global":origin["native"],"native_frozen":origin["native"]}
    treatments={}
    physical_cache={}
    try:
        for mode,prior in modes.items():
            history=[]
            results=[]
            for t,(s,k) in enumerate(zip(seeds,stations,strict=True)):
                arena,server=arenas[k]
                obs=arena.reset(seed=s,split="held",regime=regime)
                x=descriptor(obs,mu)
                # Smoothness on the parameter axis supports neighbouring coordinate conventions.
                pref=np.exp(-.5*((VALUES-prior)/.32)**2)
                value=1.6+1.8*pref
                mass=np.full(3,1.3)
                total=value*mass
                for old_x,old_choice,score in history:
                    if mode=="native_global":
                        locality=1.
                    else:
                        locality=np.exp(-.5*np.sum((x-old_x)**2)/6.0)
                    influence=locality*np.exp(-.5*((VALUES-VALUES[old_choice])/.26)**2)
                    mass+=influence
                    total+=influence*min(score,15)
                if mode=="native_frozen":
                    alpha=prior
                    choice=int(np.argmin(abs(VALUES-alpha)))
                else:
                    # Optimism pays for trying a different convention after an unexpected low return.
                    bonus=3.8*np.sqrt(np.log(2+len(history))/mass)
                    choice=int(np.argmax(total/mass+bonus))
                    alpha=float(VALUES[choice])
                if (t,alpha) not in physical_cache:
                    physical_cache[t,alpha]=episode(arena,server,s,native,w,mu,"scalar",regime,alpha=alpha,
                                                    lead_override=0 if k and immediate else None)
                # Deterministic duplicate choices can reuse the exact on-policy tape on the identical start.
                row=physical_cache[t,alpha]
                history.append((x,choice,row["rebounds"]))
                results.append(row|dict(station=k,alpha=alpha,context=x.tolist()))
            treatments[mode]=results
            print(index,mode,[sum(r["run"] for r in results[a:b]) for a,b in ((0,5),(5,12),(12,16),(16,20))],flush=True)
        with gzip.open(ROOT/f"station-switch-{'immediate' if immediate else 'held'}-stream-{index}.json.gz","wt") as f:
            json.dump(dict(seeds=seeds,body=body,modes=treatments,unique_physical_episodes=len(physical_cache)),f,separators=(",",":"))
    finally:
        for arena,server in arenas:
            arena.close()
            server.close()
        native.working.cleanup()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen",action="store_true")
    p.add_argument("--immediate",action="store_true")
    cfg=p.parse_args()
    if cfg.screen:
        screening(cfg.immediate)
    else:
        with ProcessPoolExecutor(max_workers=3) as pool:
            list(pool.map(stream,[(i,cfg.immediate) for i in range(6)]))
