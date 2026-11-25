"""Reserve ordinary stock losses with no known successful endpoint."""

import argparse
from contextlib import ExitStack
from hashlib import blake2b
from pathlib import Path

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt53_signed_practice as infer
from .tt4_transfer_trial import _hash, _read, _write
from .tt44_paired_credit import prefix
from .tt48_rim_recovery import STOCK

PLAN=infer.ROOT/"fresh.protocol.json"
OUT=infer.ROOT/"fresh.json.gz"
FIRST=8


def plan(out):
    reference=_read(STOCK)
    trainer=_read(infer.LINES)
    old={r["id"] for r in trainer["cases"]}
    cards=[]
    for i,(c,r) in enumerate(zip(reference["cards"][:FIRST],reference["rows"][:FIRST],strict=True)):
        if c["id"] in old or c["opening"]!="random" or c["profile"]!="reach":
            raise ValueError("Fresh release overlaps movement donors or changes opponent")
        delivery=r["placements"]["delivered"]
        leg=len(delivery)-1
        record=dict(index=i,id=c["id"],card=c)
        if not delivery or delivery[-1]["near_landed"]:
            cards.append(dict(record,eligible=False,reason="No final failed delivered roll"))
            continue
        start=delivery[-1]["far_contact"]["time"]
        look=[n for n in r["searches"] if n["tick"]*mpc.DT>start and -1.30<n["p"][1]<-.50]
        if not look:
            cards.append(dict(record,eligible=False,reason="No public search state on the final failed leg"))
            continue
        tick=int(look[0]["tick"])
        u=np.ascontiguousarray(r["actions"][:tick],dtype="<f8")
        cards.append(dict(record,eligible=True,tick=tick,leg=leg,
                          prefix=blake2b(u.tobytes(),digest_size=16).hexdigest()))
    _write(out,dict(ordered=cards,stock_sha256=_hash(STOCK),basis_sha256=_hash(infer.LINES),
                    learner_sha256=_hash(Path(infer.__file__)),code_sha256=_hash(Path(__file__)),
                    root="First eight fixed independent official assessment placements before examining any current correction. At most one terminal failed bounced delivery per game. The decision is the first ordinary public optical search with depth p_y in [-1.30,-.50] on that leg. No successful endpoint or actual contact time is recorded in the plan. A failed replay, four bounded practice attempts and one equal-budget final per arm at exactly the same native incoming offer."))


def read_position(a,cam,rig,row,card,tick):
    prev=prefix(a,card,row,tick,cam,a._arm_ids[1][2],window=26)
    eye=mpc.Vision(rig,a.control_dt,raw=True)
    eye.eye=mpc.LearnedOptics(eye.eye,course.OPTIC)
    for k,rgb,q,_ in prev[-26:]:
        mpc.observed(eye,k,rgb,q)
    return mpc.observed(eye,tick,cam.capture() if tick%2==0 else None,a.joints)


def run(protocol,out):
    source,reference,training=_read(protocol),_read(STOCK),_read(infer.LINES)
    if (source["stock_sha256"]!=_hash(STOCK) or source["basis_sha256"]!=_hash(infer.LINES)
            or source["learner_sha256"]!=_hash(Path(infer.__file__)) or source["code_sha256"]!=_hash(Path(__file__))):
        raise ValueError("Fresh trial not frozen")
    result=dict(protocol_sha256=_hash(protocol),rows=[],complete=False)
    if out.exists():
        result=_read(out)
        if result["protocol_sha256"]!=_hash(protocol):
            raise ValueError("Fresh continuation changed")
    cal=_read(infer.CAMERA)["models"]["60"]
    with mpc.centered(reference["ready"]),ExitStack() as stack:
        a=mpc.arena();stack.callback(a.close)
        cam,rig,stride,lo,hi=mpc.resources(a);stack.callback(cam.close)
        for stamp in source["ordered"][len(result["rows"]):]:
            i=stamp["index"]
            c,r=reference["cards"][i],reference["rows"][i]
            if c!=stamp["card"]:
                raise ValueError("Changed fresh opponent")
            if not stamp["eligible"]:
                result["rows"].append(stamp)
            else:
                t=stamp["tick"]
                seen=read_position(a,cam,rig,r,c,t)
                if seen is None:
                    v=dict(stamp,native_sensor_missing=True)
                else:
                    line=dict(stamp,observed=seen.tolist())
                    other=infer.source_bank(training["cases"],c["id"])
                    v=infer.test(stamp,line,other,r,c,a,cam,rig,stride,lo,hi,cal)
                    v.update(stamp)
                result["rows"].append(v)
                print(c["id"],"shared",v.get("common_practice_rescue"),"signed",v.get("decision",{}).get("signed"),
                      {k:e["scored"] for k,e in v.get("arms",{}).items()},flush=True)
            result["complete"]=len(result["rows"])==len(source["ordered"])
            _write(out,result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","run"))
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args()
    plan(a.plan) if a.mode=="plan" else run(a.plan,a.out)
