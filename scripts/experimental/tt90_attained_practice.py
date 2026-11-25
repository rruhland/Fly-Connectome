"""Acquire scored direct strokes from actually attained bodies in independent TT90 practice."""

from collections import Counter
from unittest.mock import patch

import numpy as np

from . import tt90_installation_learning as tt
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt40_long_low import same_trace
from .tt44_paired_credit import State

OUT=tt.ROOT/"coached.json.gz"
QUOTA={"low":8,"wing":5,"other":3}


def category(offer):
    line=offer["receive_line"]; z=tt.optics.projected_height(line); x=line["xyz"][0]
    if z<1.02:return "low"
    if abs(x)>.27:return "wing"
    return "other"


def select(d):
    picked=[]; counts=Counter()
    for j,(c,r) in enumerate(zip(d["cards"],d["rows"])):
        offers=r["placements"]["delivered"]
        if not offers or offers[-1]["near_landed"] or not offers[-1]["receive_line"] or not offers[-1]["near_bounce"]:continue
        kind=category(offers[-1])
        if counts[kind]<QUOTA[kind]:
            picked.append(dict(index=j,id=c["id"],leg=len(offers)-1,kind=kind))
            counts[kind]+=1
    return picked


def seed_state(a,c,r,leg):
    start=r["placements"]["delivered"][leg]["far_contact"]["time"]
    end=r["placements"]["delivered"][leg]["near_bounce"]["time"]
    e=next(e for e in r["events"] if e.get("collider")=="rival_racket" and start<=e["time"]<end)
    j=e["step"]+1
    a.launch(r["shot"],card=c); a.partner.reset(seed=c["seed"]); a.sample_clearance(False)
    prefix=r["actions"][:j]
    for u in prefix:a.exchange(u,tt.BASE.FAR_COMMAND)
    if digest_json(a.events)!=digest_json([e for e in r["events"] if e["time"]<=a.data.time+1e-10]):
        raise ValueError("Changed actual near-body prefix before physical practice")
    return State(a),prefix


def traces(a,cam,rig,ready,c,shot,r,leg):
    a.launch(shot,card=c);a.partner.reset(seed=c["seed"])
    eye=tt.BASE.Vision(rig,a.control_dt,raw=True);eye.eye=tt.BASE.LearnedOptics(eye.eye,tt.COURSE.OPTIC)
    controls=r["controls"]; stride,lo,hi=tt.library.limits(a)
    ref=tt.library.references(controls,ready,stride,lo,hi)
    observed=[];when=r["offer"]["near_contact"]["step"]
    for t,u in enumerate(controls):
        q=a.joints
        x=tt.BASE.observed(eye,t,cam.capture() if t%2==0 and t>=when-52 else None,q)
        if when-26<=t<=when and x is not None and x[4]<-.90 and -1.78<x[1]<-.37:
            observed.append([t,x.tolist(),q.tolist(),ref[t].tolist(),ref[t+1].tolist()])
        a.exchange(u,tt.BASE.FAR_COMMAND)
    if digest_json(a.events)!=digest_json(r["events"]):
        raise ValueError("Acquiring RGB/encoder labels changed actual coached action evidence")
    return dict(actions=controls, score=dict(learner=sum(a["near_landed"] for a in r["placements"]["delivered"])),
                placements=r["placements"],optical_earned=[dict(tag=["coached",c["id"],leg+1],frames=observed)],
                optical_missed=[])


def teach(a,guard,c,r,leg):
    snapshot,prefix=seed_state(a,c,r,leg); original=r["placements"]["delivered"][leg]
    coach=tt.school.physical.low.Probe(a); base=coach.basis()
    with patch.object(tt.school.physical,"STAGE_TRIALS",104),patch.object(tt.school.physical,"ROUNDS",3):
        a.clearance_margin=.0008
        try:candidates,stats=tt.school.physical.search(coach,[903093,c["seed"],leg],base)
        finally:a.clearance_margin=0.
    attempts=[]
    for u,meta in candidates[:3]:
        v=tt.school.trial(a,snapshot,c,prefix,leg,u+[tt.school.ZERO]*155,guard)
        received=v["offer"]; v["forecast"]=meta
        v["same"]=bool(received and all(same_trace(received[k],original[k]) for k in ("far_contact","near_bounce")))
        attempts.append(v)
        if tt.school.physical.received(v):
            v["strict"]=tt.school.verify(a,c,r["shot"],v,guard)
            break
    return dict(stats=stats,attempts=attempts,basis=len(base))


def run(feeds):
    p=tt.contract(feeds); d=_read(tt.path("practice")); m,ready,_=tt.material()
    if len(d["rows"])!=len(p["learn"]):raise ValueError("Complete the independent broad native practice before resampling misses")
    plan=select(d)
    report=dict(practice=_hash(tt.path("practice")),protocol=_hash(tt.PLAN),cases=plan,rows=[],quota=QUOTA,complete=False)
    if OUT.exists():
        x=_read(OUT)
        if any(x[k]!=v for k,v in report.items() if k not in ("rows","complete")):raise ValueError("Changed mounted attainments")
        report=x
    with tt.school.enabled(),tt.system.installed(m["v"],m["mount"]),tt.BASE.centered(ready):
        a=tt.BASE.arena(); cad=tt.BASE.arena(width=64,height=48,detail="cad"); cam,rig,*_=tt.BASE.resources(a)
        guard=tt.school.physical.GripClearance(cad.model,stock=True)
        try:
            for item in plan[len(report["rows"]):]:
                j=item["index"];c=d["cards"][j];r=d["rows"][j]
                v=teach(a,guard,c,r,item["leg"])
                earned=next((w for w in v["attempts"] if w.get("strict",{}).get("penetration_samples")==0),None)
                v["sensory"]=None if earned is None else traces(a,cam,rig,ready,c,r["shot"],earned,item["leg"])
                report["rows"].append(v);report["complete"]=len(report["rows"])==len(plan);_write(OUT,report)
                print("attained",item["kind"],c["id"],item["leg"],None if earned is None else round(1000*earned["contact"]["radial_m"],2),flush=True)
            if not plan:report["complete"]=True;_write(OUT,report)
        finally:cam.close();cad.close();a.close()
