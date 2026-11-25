"""Read off provenance, cross-game credit support, and exact physical paired outcomes."""

import argparse
from ast import literal_eval
from collections import Counter
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt52_delayed_motor_credit import GAMMA, OUT, PLAN, ROOT, START, STOCK, is_live, sham_values


def archive_support(protocol):
    """Held-game observational test of the same state/motor neighborhood."""
    source=START.with_suffix("").with_suffix(".npz")
    values={(g,s):(a+GAMMA*b)/(1.+GAMMA*bool(b)) for g,s,a,b in protocol["start_credit"] if a}
    sham=sham_values(values)
    with np.load(source) as z:
        states=z["states"]
        targets=z["goals"][:,[3,9,16],:5].reshape(-1,15)
        keys=list(map(literal_eval,z["tags"]))
    legs={}
    for i,k in enumerate(keys):
        if is_live(k) and k[1:] in values:
            legs.setdefault(k[1:],[]).append(i)
    k=sorted(legs)
    middle=[legs[t][len(legs[t])//2] for t in k]
    x,u=states[middle],targets[middle]
    y=np.asarray([values[t] for t in k])
    yep=np.asarray([sham[t] for t in k])
    groups=np.asarray([g for g,s in k])
    g,idx=np.unique(groups,return_inverse=True)
    G=np.eye(len(g))[idx]
    d=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=-1)
    scope=(d<2.65)&(groups[:,None]!=groups[None,:])
    # One representative frame per earned action, no same-game links.
    w=np.exp(-.5*(d/1.16)**2)*scope
    w/=np.maximum(1.,(w@G)[:,idx])
    weight=w.sum(axis=1)
    prior=(w@y+1.5*y.mean())/(weight+1.5)
    sham_prior=(w@yep+1.5*y.mean())/(weight+1.5)
    action=np.exp(-.5*np.mean((u[:,None,:]-u[None,:,:])**2,axis=-1)/.19**2)
    both=w*action
    used=(weight>1.25)&(both.sum(axis=1)>.85)&((both@G>.16).sum(axis=1)>=2)
    estimate=(both@y+1.25*prior)/(both.sum(axis=1)+1.25)
    sham_est=(both@yep+1.25*sham_prior)/(both.sum(axis=1)+1.25)
    result=dict(n=len(y),games=len(g),qualified_n=int(used.sum()),
                distribution=dict(Counter(map(str,y))),support_rule="At least two different games; no current game's outcomes or hidden ball state; same matched distances and sham per-game labels.")
    for kind,h in (("all",weight>0),("qualified",used)):
        if not np.any(h):
            continue
        result[kind]={name:float(np.mean((v[h]-y[h])**2)) for name,v in
                      (("archived_prevalence",np.full(len(y),y.mean())),("state_only",prior),
                       ("same_state_plus_motor",estimate),("same_state_plus_motor_sham",sham_est))}
    return result


def summarize(run,protocol):
    ref=_read(STOCK)
    if _hash(STOCK)!=protocol["stock_sha256"]:
        raise ValueError("Reference changed")
    rows=run["rows"]
    s=np.array([r["score"]["learner"] for r in ref["rows"][:len(rows)]],int)
    y=np.array([r["score"]["learner"] for r in rows],int)
    stats=sum((Counter(r["credit_telemetry"]) for r in rows),Counter())
    gains=y-s
    res=dict(cards=[r["id"] for r in rows],matched=len(y),complete=run["complete"],
             planned_n=protocol["n"],score=[int(s.sum()),int(y.sum())],
             best=[int(s.max(initial=0)),int(y.max(initial=0))],
             wins=int((gains>0).sum()),ties=int((gains==0).sum()),losses=int((gains<0).sum()),
             delta=gains.tolist(),earliest_differences=[r["first_different_action"] for r in rows],
             telemetry=dict(stats),
             opponent_boundary="Same measured opening, independent deterministic broad far target draws reused across paired games. Native replies and attained states can differ after the first changed issued command.")
    if len(gains)>1:
        ix=np.random.default_rng(52034).integers(len(gains),size=(4000,len(gains)))
        res["mean_return_difference_ci95"]=np.quantile(gains[ix].mean(axis=1),[.025,.975]).tolist()
    return res


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--run",type=Path,default=OUT)
    p.add_argument("--out",type=Path,default=ROOT/"report.json")
    a=p.parse_args()
    protocol=_read(a.plan)
    summary=dict(protocol_sha256=_hash(a.plan),archive_support=archive_support(protocol))
    if a.run.exists():
        r=_read(a.run)
        if r["plan_sha256"]!=summary["protocol_sha256"]:
            raise ValueError("Physical report and schedule mismatch")
        summary["paired"]=summarize(r,protocol)
        summary["tape_sha256"]=_hash(a.run)
    _write(a.out,summary)
    print(summary,flush=True)
