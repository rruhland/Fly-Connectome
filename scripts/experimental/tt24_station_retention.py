"""Separate learning from grid rounding and evaluate stored station preferences on new plays."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json

import numpy as np

from .tt24_broad_donors import fixed_seeds
from .tt24_donor_trial import Native, ROOT
from .tt24_ready import episode
from .tt24_station_switch import VALUES, course, descriptor, read_parameters


def preference(history,x,prior,local):
    pref=np.exp(-.5*((VALUES-prior)/.32)**2)
    mass=np.full(3,1.3)
    total=(1.6+1.8*pref)*mass
    for old in history:
        a=old["alpha"]
        affinity=np.exp(-.5*np.sum((x-old["context"])**2)/6.) if local else 1.
        influence=affinity*np.exp(-.5*((VALUES-a)/.26)**2)
        mass+=influence
        total+=influence*min(old["rebounds"],15)
    # Evaluate the learned expected value without an exploration bonus.
    return float(VALUES[np.argmax(total/mass)])


def run(index):
    with gzip.open(ROOT/f"station-switch-immediate-stream-{index}.json.gz","rt") as f:
        report=json.load(f)
    with gzip.open(ROOT/"tt24_pong_origin-bank.json.gz","rt") as f:
        origin=json.load(f)["replicas"][index]["selected"]
    w,mu,_=read_parameters()
    native=Native()
    body=report["body"]
    regime="standard" if body=="stock" else "changed"
    arenas=[course(k,body,True) for k in (0,1)]
    try:
        prior=origin["native"]
        frozen=float(VALUES[np.argmin(abs(VALUES-prior))])
        fair=[]
        for t,row in enumerate(report["modes"]["native_memory"]):
            kind=row["station"]
            existing=next((v[t] for v in report["modes"].values() if v[t]["alpha"]==frozen),None)
            if existing is None:
                arena,server=arenas[kind]
                existing=episode(arena,server,row["seed"],native,w,mu,"scalar",regime,alpha=frozen,
                                 lead_override=0 if kind else None)
            fair.append(dict(seed=row["seed"],run=existing["run"],rebounds=existing["rebounds"]))
        seeds=fixed_seeds(8,181401132+index)
        held=[]
        for t,s in enumerate(seeds):
            k=t%2
            arena,server=arenas[k]
            x=descriptor(arena.reset(seed=s,split="held",regime=regime),mu)
            choices={"frozen":frozen}
            for name,base in (("native_memory",prior),("shuffled_memory",origin["shuffled"]),
                              ("cold_memory",.5),("native_global",prior)):
                choices[name]=preference(report["modes"][name],x,base,name!="native_global")
            played={}
            for a in dict.fromkeys(choices.values()):
                played[str(a)]=episode(arena,server,s,native,w,mu,"scalar",regime,alpha=a,lead_override=0 if k else None)
            held.append(dict(seed=s,station=k,choices=choices,played=played))
        with gzip.open(ROOT/f"station-revisit-{index}.json.gz","wt") as f:
            json.dump(dict(body=body,rounded_prior=frozen,live_frozen_rounded=fair,held=held),f,separators=(",",":"))
    finally:
        for arena,server in arenas:
            arena.close()
            server.close()
        native.working.cleanup()


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(run,range(6)))
