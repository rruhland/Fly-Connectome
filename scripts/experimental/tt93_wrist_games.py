"""Run bounded fresh matched full points only after the changed wrist cue earns admission."""
import argparse
from collections import deque
from dataclasses import replace
from time import process_time

import numpy as np

from . import tt93_wrist_course as course
from . import tt93_open_wrist_fixture as mount
from .tt37_opponent_audit import CourtClearance, guarded
from .tt29_endurance import audit
from .tt19_view_full_game import digest_json
from .tt40_long_low import same_trace

mpc,school=course.mpc,course.school
_read,_write,_hash=course._read,course._write,course._hash
OUT=mount.ROOT/"games.json.gz"
SUMMARY=mount.ROOT/"game-summary.json"
COUNT=4


def play(a,res,actor,home,entry,fuse,*,live=False,progress=0,on_hundred=None):
    cam,rig,stride,lower,upper=res
    c=entry["card"]
    vision=mpc.Vision(rig,a.control_dt,raw=True)
    vision.eye=mpc.LearnedOptics(vision.eye,course.course.OPTIC)
    actor.begin()
    a.launch(entry["shot"],card=c);a.partner.reset(seed=c["seed"])
    ref=np.asarray(home).copy()
    moves=[]; corrections=[]; inbound=False
    previous_reward=0;offer=False
    before=actor.priors.digest();recent=deque(maxlen=150)
    start_cpu=process_time()
    for j in range(a.max_steps):
        q=a.joints.copy()
        x=mpc.observed(vision,j,cam.capture() if j%2==0 else None,q)
        if x is not None and x[4]<-.9 and x[1]<-.9:
            inbound=True
        raw=vision.eye.latest
        if fuse and x is not None and x[4]<-.9 and -.16<x[5] and -1.50<x[1]<-.53 and .78<x[2]<1.18:
            lens=rig.wrist.at(q)
            pred=lens.project(x[:3])
            if pred[2]>.05 and np.all(pred[:2]>6) and np.all(pred[:2]<[lens.width-7,lens.height-7]):
                rgb=a.render_wrist()
                sight=course.native_ball_pixels(rgb,pred[:2])
                delta=course.correction(lens,sight,x[:3])
                if delta is not None:
                    x=np.r_[x[:3]+delta,x[3:]]
                    vision.eye.latest=replace(raw,position=x[:3],velocity=x[3:],time=j*a.control_dt)
                    corrections.append(dict(tick=j,delta_m=delta.tolist()))
        start=np.asarray(home)
        if x is not None and (x[4]<-.9 or inbound and x[1]<.50):
            quote,conf=actor.raw.get(x,q)
            if quote is not None:
                start=home+conf*(quote-home)
        target=actor.step(j,vision.eye,x,q,ref,start)
        # The learned optical history past this tick remains calibrated on
        # standard stereo. Wrist correction is a causal near-field update.
        vision.eye.latest=raw
        u=np.clip((np.clip(target,lower,upper)-ref)/stride,-.8,.8); sent=np.clip(ref+u*stride,lower,upper)
        _,_,done,cap,info=a.exchange(u,mpc.FAR_COMMAND)
        moves.append(u.tolist())
        if live:
            recent.append((j,x,q,ref,sent))
            for e in info["events"]:
                if e["kind"]=="legal_return":
                    offer=e["side"]==-1
            if any(e["kind"]=="legal_return" and e["side"]==1 for e in info["events"]):
                reward=audit(a.events,opening_side=1)["learner"]
                if reward>previous_reward:
                    kept=actor.priors.keep(list(recent),("live",actor.serial,reward))
                    actor.diag["credited_fragments"]+=kept
                    actor.diag["updates"]+=bool(kept)
                    if on_hundred is not None and previous_reward<100<=reward:
                        on_hundred(dict(actions=moves,events=a.events,placements=a.point_placements(),
                                      score=audit(a.events,opening_side=1),complete_point=False,
                                      corrections=corrections))
                    previous_reward=reward
                    if progress and reward%progress==0:
                        print("TT93 in point",c["id"],"fused" if fuse else "stereo",reward,j,flush=True)
        ref=sent
        if done or cap:break
    if live and done and offer and hasattr(actor,"missed"):
        actor.missed(list(recent))
    place=a.point_placements()
    for leg,row in enumerate(place["requested"]):
        if row["target"]!=list(course.draw(c,leg,"random",None)[0]):
            raise ValueError("Native goal law changed")
    return dict(score=audit(a.events,opening_side=1),events=a.events,placements=place,actions=moves,
                corrections=corrections,complete_point=a.events[-1]["kind"]=="point",
                memory=dict(before=before,after=actor.priors.digest()),planner=dict(actor.diag),
                runtime_cpu_s=process_time()-start_cpu)


def verify(a,guard,entry,match):
    a.launch(entry["shot"],card=entry["card"]);a.partner.reset(seed=entry["card"]["seed"])
    guarded(a,guard)
    enclosure=mount.Clear(a)
    samples=0
    block=[]
    for t,u in enumerate(match["actions"]):
        if max(abs(np.asarray(u)))>.8+1.e-10:
            raise ValueError("Direct rated action exceeded bounds")
        q=a.joints.copy(); faults=enclosure.at(q);samples+=1
        if faults:block.append((t,faults[:2]))
        a.exchange(u,mpc.FAR_COMMAND)
    if digest_json(a.events)!=digest_json(match["events"]) or digest_json(a.point_placements())!=digest_json(match["placements"]):
        raise ValueError("Replayed game not physically exact")
    c=a.clearance_result()
    if c["samples"]!=round(a.data.time/a.model.opt.timestep):
        raise ValueError("Clearance samples omitted")
    return dict(global_cad=c,local_module_samples=samples,local_module_penetrations=block)


def games():
    r=_read(course.REPORT)
    if not r["admit_prospective_fusion"]:
        _write(OUT,dict(complete=True,rows=[],status="The physical and optical battery did not support a changed-sensor game.",
                        optical_sha256=_hash(course.REPORT)))
        return
    p,m=course.guaranteed()
    items=p["future_full_points"][:COUNT]
    ctx=dict(complete=False,rows=[],optical_sha256=_hash(course.REPORT),model_sha256=_hash(mount.MODEL),
             plan_sha256=_hash(course.PLAN),scope="Both controllers use the same truly installed TT93 carrier and mass.",
             control="Same frozen stock RGB receiver and TT37 draws with or without a conservative transverse wrist update.")
    if OUT.exists():
        d=_read(OUT)
        if any(d[k]!=ctx[k] for k in ("model_sha256","plan_sha256","optical_sha256")):
            raise ValueError("Not the same sealed full-game exercise")
        ctx=d
    home,raw,rank,cp=course.act_setup();ids={r["card"]["id"] for r in items}
    with mount.builder(m),mpc.centered(home),school.enabled():
        a=mpc.arena(); proof=mpc.arena(width=64,height=48)
        cad=mpc.arena(width=64,height=48,detail="cad")
        guard=CourtClearance(cad.model,articulated=True)
        res=mpc.resources(a)
        try:
            for item in items[len(ctx["rows"]):]:
                arms={}
                for arm,fuse in (("same-installation-stereo",False),("same-installation-wrist-fusion",True)):
                    ctl=course.actor_for(raw,rank,cp,ids)
                    try:
                        record=play(a,res,ctl,home,item,fuse)
                        record["cad"]=verify(proof,guard,item,record)
                        arms[arm]=record
                        print(item["card"]["id"],arm,record["score"]["learner"],len(record["corrections"]),
                              record["complete_point"],record["cad"]["global_cad"]["penetration_samples"],
                              len(record["cad"]["local_module_penetrations"]),flush=True)
                    finally:ctl.close()
                ctx["rows"].append(dict(item,arms=arms));_write(OUT,ctx)
        finally:
            res[0].close();a.close();proof.close();cad.close()
    ctx["complete"]=True;_write(OUT,ctx)


def summarize():
    by=_read(OUT)
    if not by["complete"]:
        raise ValueError("All matched native terminal scores are required")
    s=[]
    for r in by["rows"]:
        old,new=(r["arms"][k] for k in ("same-installation-stereo","same-installation-wrist-fusion"))
        z=[]
        for i,(a,b) in enumerate(zip(old["placements"]["delivered"],new["placements"]["delivered"])):
            if all(same_trace(a[k],b[k]) for k in ("far_contact","near_bounce")):
                z.append(dict(leg=i,before=a["near_landed"],after=b["near_landed"]))
        i=next((i for i,(a,b) in enumerate(zip(old["actions"],new["actions"])) if np.max(np.abs(np.array(a)-b))>1.e-8),None)
        s.append(dict(id=r["card"]["id"],stock=old["score"]["learner"],fused=new["score"]["learner"],
                      applied=len(new["corrections"]),first_action_difference=i,
                      terminal=[a["complete_point"] for a in (old,new)],matched_offers=z,
                      global_cad_samples=[a["cad"]["global_cad"]["samples"] for a in (old,new)],
                      global_penetrations=[a["cad"]["global_cad"]["penetration_samples"] for a in (old,new)],
                      local_penetrations=[len(a["cad"]["local_module_penetrations"]) for a in (old,new)]))
    _write(SUMMARY,dict(source_sha256=_hash(OUT),rows=s,
              stock=sum(r["stock"] for r in s),fused=sum(r["fused"] for r in s)))
    print("TT93 complete full-game totals",sum(r["stock"] for r in s),sum(r["fused"] for r in s),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("games","summary"),nargs="?",default="games")
    {"games":games,"summary":summarize}[p.parse_args().mode]()
