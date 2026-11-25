"""Ask whether the four learned probes beat exactly rate-matched scrambled coordination."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path

import numpy as np

from . import tt53_signed_practice as practice
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt44_paired_credit import State, prefix

PLAN=practice.ROOT/"geometry.protocol.json"
OUT=practice.ROOT/"geometry.json.gz"
TRIAL=practice.ROOT/"trial.json.gz"
STREAM="tt53-independent-actuator-signs-20261125"


def signs(key):
    """Independently scramble the relative joint signs of each time path."""
    masks=[]
    for axis in range(2):
        h=sha256(f"{STREAM}/{key}/{axis}".encode()).digest()
        # Exclude global plus and minus. They are identical when the full
        # antithetic pair is tested and do not change coordination.
        ident=1+int.from_bytes(h[:8],"little")%30
        masks.append([1 if ident&(1<<j) else -1 for j in range(5)])
    return np.asarray(masks)


def controls(failed,basis,stride,lo,hi,ref):
    return np.asarray([practice.issue(failed,basis,practice.ALPHA*np.asarray(direction),
                                     stride,lo,hi,ref) for _,direction in practice.REPEATS])


def equal_marginals(native,sham):
    # Actuator filtering is separable. After nonlinear clipping and slew,
    # every frame still has the identical multiset of issued u on each
    # individual axis. The differences are solely joint combinations.
    return np.array_equal(np.sort(native,axis=0),np.sort(sham,axis=0))


def schedule(out):
    tape=_read(TRIAL)
    if not tape["complete"] or tape["n"]!=17 or len(tape["rows"])!=17:
        raise ValueError("Expected the complete pinned probe test")
    _write(out,dict(source_sha256=_hash(TRIAL),pairs_sha256=_hash(practice.PAIRS),lines_sha256=_hash(practice.LINES),
                    code_sha256=_hash(Path(__file__)),signed_practice_sha256=_hash(Path(practice.__file__)),
                    root=STREAM, roster=[dict(id=r["id"],tick=r["tick"],prefix=r["prefix"],
                     signs=signs(r["prefix"]).tolist()) for r in tape["rows"]],
                    rule="Compare the same four native antithetic probes against the same four probes whose entire joint axes receive independent fixed sign flips. Individual action distributions are verified equal at every tick even after clipping and slew, but coordination is scrambled. Same exact incoming state and live broad-opponent stream, no geometric oracle or RGB post-action choice."))


def sham_roll(a,snap,old,leg,u):
    snap.restore(a)
    t0=len(a.events)
    steps=[]
    reward=False
    for motor in u:
        _,_,done,capped,info=a.exchange(motor,practice.mpc.FAR_COMMAND)
        steps.append(motor.tolist())
        reward=any(e["kind"]=="legal_return" and e.get("side")==1 for e in info["events"])
        if reward or done or capped:
            break
    delivery=a.point_placements()["delivered"]
    original=old["placements"]["delivered"][leg]
    matched=len(delivery)>leg and all(digest_json(delivery[leg][k])==digest_json(original[k])
                                      for k in ("far_contact","near_bounce"))
    return dict(scored=bool(reward and matched),matched=bool(matched),actual_legal=bool(reward),
                controls=steps,events=a.events[t0:])


def run(protocol,out):
    plan,tape,pairs,lines=map(_read,(protocol,TRIAL,practice.PAIRS,practice.LINES))
    expect={"source_sha256":TRIAL,"pairs_sha256":practice.PAIRS,"lines_sha256":practice.LINES,
            "code_sha256":Path(__file__),"signed_practice_sha256":Path(practice.__file__)}
    if any(plan[k]!=_hash(p) for k,p in expect.items()) or not tape["complete"]:
        raise ValueError("Changed sham geometry comparison")
    data=dict(protocol_sha256=_hash(protocol),rows=[],complete=False)
    if out.exists():
        old=_read(out)
        if old["protocol_sha256"]!=data["protocol_sha256"]:
            raise ValueError("Cannot resume a different basis sham")
        data=old
    todo=zip(plan["roster"],tape["rows"],pairs["pairs"],lines["cases"],strict=True)
    stack=ExitStack()
    cache={}
    ready=None
    try:
        for i,(fixed,row,pair,line) in enumerate(todo):
            if i<len(data["rows"]):
                continue
            fields=("id","tick","prefix")
            if any(len({x[k] for x in (fixed,row,pair,line)})!=1 for k in fields):
                raise ValueError("Changed decision fork roster")
            picked,doc=practice.stock_endpoint(pair,line,cache)
            if _hash(Path(picked["source"]))!=pairs["sources"][picked["source"]]:
                raise ValueError("Changed actual failed source")
            old,card=doc["rows"][picked["row"]],doc["cards"][picked["row"]]
            if doc["ready"]!=ready:
                stack.close()
                ready=doc["ready"]
                stack.enter_context(practice.mpc.centered(ready))
                a=practice.mpc.arena()
                stack.callback(a.close)
            actual_axes=a._arm_ids[1][2]
            stride=a._joint_increments[actual_axes]*(a.control_dt/a.model.opt.timestep)
            lo,hi=a._joint_low[actual_axes],a._joint_high[actual_axes]
            prefix(a,card,old,row["tick"],None,actual_axes,window=0)
            snap=State(a)
            ref=a.data.ctrl[actual_axes].copy()
            failure=np.asarray(old["actions"][row["tick"]:row["tick"]+100])
            failure=np.pad(failure,((0,100-len(failure)),(0,0)))
            basis=np.asarray(row["basis"])
            mask=np.asarray(fixed["signs"])
            if not np.array_equal(mask,signs(row["prefix"])):
                raise ValueError("Changed independent sham basis")
            native=controls(failure,basis,stride,lo,hi,ref)
            random_basis=basis*mask[:,None,:]
            sham=controls(failure,random_basis,stride,lo,hi,ref)
            if not equal_marginals(native,sham):
                raise ValueError("The per-tick issued interventions changed")
            score=[]
            for u,(k,_) in zip(native,practice.REPEATS,strict=True):
                prev=row["practice"][k]
                v=np.asarray(prev["controls"])
                if not np.array_equal(v,u[:len(v)]) or not prev["matched_far_and_bounce"]:
                    raise ValueError("Reconstructed learned probe differs from fixed native tape")
                score.append(prev["scored"])
            other=[sham_roll(a,snap,old,picked["leg"],u) for u in sham]
            data["rows"].append(dict(**{k:row[k] for k in fields},sham_signs=mask.tolist(),
                                    exact_per_axis_marginals=True,learned_legal=score,sham=other))
            data["complete"]=len(data["rows"])==len(tape["rows"])
            _write(out,data)
            print(i,row["id"],sum(score),sum(e["scored"] for e in other),
                  sum(e["matched"] for e in other),flush=True)
    finally:
        stack.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","run"))
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args()
    schedule(a.plan) if a.mode=="plan" else run(a.plan,a.out)
