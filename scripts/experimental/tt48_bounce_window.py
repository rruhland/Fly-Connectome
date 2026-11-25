"""Measure how many genuinely new pixels a stock arm can wait for before a rim repair."""

import argparse
from pathlib import Path

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent_audit import CourtClearance
from .tt44_paired_credit import State
from .tt48_contact_patch import ROOT, suffix
from .tt48_contact_optical_check import CHECKS


def catchup(start, stride, lo, hi, stock, earned, wait):
    """Keep the stock prefix, then track the already earned bounded references."""
    q=start.copy()
    q0=q.copy()
    result=[]
    n=max(len(stock),len(earned))
    for i in range(n):
        delta=np.asarray(earned[i]) if i<len(earned) else np.zeros(len(q))
        q0=np.clip(q0+stride*delta,lo,hi)
        raw=(q0-q)/stride
        action=np.asarray(stock[i]) if i<min(wait,len(stock)) else np.clip(raw,-.8,.8)
        if np.any(abs(action)>.800000001):
            raise ValueError("Bounded direct command changed")
        q=np.clip(q+stride*action,lo,hi)
        result.append(action.tolist())
    return result


def probe(ident,source,witness):
    archive,receipt=_read(source),_read(witness)
    if _hash(source)!=receipt["source_sha256"]:
        raise ValueError("Changed causal proof")
    row=next(r for r in receipt["rows"] if r["id"]==ident)
    old,card=archive["rows"][row["index"]],row["card"]
    fork=row["fork"]
    event=old["placements"]["delivered"][row["leg"]]["near_bounce"]
    t=event["time"]
    # Schedule is selected for an evaluator clock bracket, not based on
    # which action scored after the bounce. The actor never receives it.
    first=int(np.ceil(t/mpc.DT-1.e-7))
    with mpc.centered(archive["ready"]):
        a=mpc.arena()
        cad=mpc.arena(detail="cad")
        try:
            a.launch(old["shot"],card=card)
            a.partner.reset(seed=card["seed"])
            for cmd in old["actions"][:fork]:
                a.exchange(cmd,mpc.FAR_COMMAND)
            captured=State(a)
            rfound=[e for e in old["events"] if e["time"]<=a.data.time+1e-8]
            if digest_json(a.events)!=digest_json(rfound):
                raise ValueError("Changed native start")
            ax=a._arm_ids[1][2]
            stride=a._joint_increments[ax]*a.control_dt/a.model.opt.timestep
            lo,hi=a._joint_low[ax],a._joint_high[ax]
            q=a.data.ctrl[ax].copy()
            controls={k:row["arms"][k]["controls"] for k in ("stock","edge")}
            guard=CourtClearance(cad.model,articulated=True)
            trials={}
            for mode,wait in (("edge_exact",0),("first_pair_after",first-fork),
                              ("two_pairs_after",first-fork+1),("three_pairs_after",first-fork+2),
                              ("four_pairs_after",first-fork+3),("six_pairs_after",first-fork+5)):
                warmed=controls["edge"] if wait==0 else catchup(q,stride,lo,hi,controls["stock"],controls["edge"],wait)
                branch=suffix(a,captured,card,old,row,warmed,clearance=guard)
                if wait==0 and branch["event_sha256"]!=row["arms"]["edge"]["event_sha256"]:
                    raise ValueError("Known immediate stroke did not replay")
                trials[mode]=dict(branch, issued_wait_ticks=wait, after_event_s=(fork+wait)*mpc.DT-t)
            return dict(id=ident, fork=fork, bounce_time_s=t, trials=trials)
        finally:
            cad.close()
            a.close()


def run(out):
    result=dict(complete=False,rows=[],sources={str(p):_hash(p) for _,a,b in CHECKS for p in (a,b)},
                scope="Privileged evaluation clock selects a stock-prefix length only. The tested direct bounded late suffix moves toward the already earned edge goal from its physically reached arm state. It does not steer with observed future motion or claim online transfer.")
    for k,a,b in CHECKS:
        r=probe(k,a,b)
        result["rows"].append(r)
        result["complete"]=len(result["rows"])==len(CHECKS)
        _write(out,result)
        print(k,[(tag,v["landed"],v["answered"],v["clearance"]["penetration_samples"])
                 for tag,v in r["trials"].items()],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT/"bounce-window.json.gz")
    run(p.parse_args().out)
