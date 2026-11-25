"""Fresh measured-launch optics and bounded fusion checks on the open TT93 wrist fixture."""
import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt93_open_wrist_fixture as mount
from . import tt85_wrist_overlap as old_look
from . import tt59_fewshot_joint_games as cm
from . import tt42_learning_course as course
from . import tt46_continuous_season as season
from .tt19_so101 import WRIST_CAMERA
from .tt31_curriculum import card,opening_candidates
from .tt29_fast_camera import native_ball_pixels
from .tt39_carry_joints import Carried
from .tt37_opponent import draw
from .tt19_view_full_game import digest_json
from .tt4_transfer_trial import _hash,_read,_write

mpc,school=mount.mpc,mount.school
ROOT=mount.ROOT
PLAN,SCOUT,HOLD,REPORT=(ROOT/n for n in ("protocol.json","scouts.json.gz","optical.json.gz","optical-summary.json"))
FEEDS=Path("/tmp/tt25-rallies.json")
SEED=9317773
DT=mpc.DT


def plan():
    train=_read(old_look.OUT)
    forbid={r["case"]["id"] for r in train["rows"]}
    for file in (mount.old_look.PLAN,Path("artifacts/tt90_installation_learning/protocol.json"),
              Path("artifacts/tt80_continuous_games/protocol.json"),Path("artifacts/tt75/protocol.json"),
              Path("artifacts/tt83_population_search/protocol.json"),Path("artifacts/tt89_partner_distillation/protocol.json")):
        if file.exists():
            forbid.update(old_look.ids(_read(file)))
    measured=mpc.measured_launches(FEEDS)
    offers=[int(s["id"]) for s in opening_candidates(measured) if s["id"] not in forbid]
    rng=np.random.default_rng(SEED)
    if len(offers)<60:
        raise ValueError("Not enough independent eligible launches")
    ids=rng.permutation(offers)[:110].tolist()
    rows=[]
    for i,r in enumerate(ids):
        c=dict(card(r,"full",i,int(rng.integers(1,2**30))),profile="reach",opening="random",
           reply_seed=int(rng.integers(1,2**30)))
        rows.append(dict(card=c,shot=measured[r]))
    ROOT.mkdir(parents=True,exist_ok=True)
    _write(PLAN,dict(schema="tt93-open-bridge-optics-1",feeds_sha=_hash(FEEDS),blocked=sorted(forbid),
        home=_read(mpc.BANK.with_suffix(".json"))["ready"],scout=rows[:-14],future_full_points=rows[-14:],
        home_memory_sha=_hash(season.CARRY),min_cue_lead_s=4*DT,max_cue_lead_s=.15,
        condition="Before any altered RGB observations, freeze a disjoint supply of fresh measured launches and standard independent TT37 streams. Scout native neutral bodies only for difficulty, not whether the optic or controller succeeds. Study the frozen hard and routine roles on the fully physically installed changed sensor, ordinary exterior-controlled motions. No stock-fixture qualification scores change."))
    print("TT93 enrolled",len(rows)-14,"actual-opening scouts and 14 possible full points",flush=True)


def guaranteed():
    p=_read(PLAN)
    m=_read(mount.MODEL)
    if p["home_memory_sha"]!=_hash(season.CARRY) or p["feeds_sha"]!=_hash(FEEDS):
        raise ValueError("Frozen inputs changed")
    if m["source"]!=_hash(old_look.OUT):
        raise ValueError("Training geometry changed")
    if not m["winner"]:
        raise ValueError("No physically supported fixture to render")
    return p,m["winner"]["spec"]


def delivery(a,leg=0):
    ds=a.point_placements()["delivered"]
    return ds[leg] if leg<len(ds) else None


def buzz(o):
    return digest_json((o["far_contact"],o["near_bounce"]))


def strata(native):
    cards=[r for r in native if r["true_deep"]]
    center=[r for r in cards if abs(r["true_deep"][0])<.118 and .935<r["true_deep"][1]<1.025]
    low=[r for r in cards if abs(r["true_deep"][0])<.286 and .945<r["true_deep"][1]<1.035 and r not in center]
    routine=[r for r in cards if abs(r["true_deep"][0])<.33 and 1.05<r["true_deep"][1]<1.16]
    return center,low,routine


def scout():
    p,m=guaranteed()
    h=p["home"]; report=dict(protocol_sha256=_hash(PLAN),model_sha256=_hash(mount.MODEL),rows=[],complete=False)
    if SCOUT.exists():
        existing=_read(SCOUT)
        if any(existing[k]!=report[k] for k in ("protocol_sha256","model_sha256")):
            raise ValueError("Cannot mix actual scouts from different sensor hardware")
        report=existing
        if report["complete"]:return
    # The pool is a maximum. The first complete nominal-geometry quota
    # closes at a twelve-card boundary, without any captured optic or score.
    bounded=p["scout"]
    for n in range(12,len(report["rows"])+1,12):
        if all(len(z)>=4 for z in strata(report["rows"][:n])):
            report["rows"]=report["rows"][:n]
            bounded=bounded[:n];break
    with mount.builder(m),mpc.centered(h),school.enabled(cameras=False):
        a=mpc.arena(width=64,height=48)
        try:
            u=np.zeros(6)
            for item in bounded[len(report["rows"]):]:
                c=item["card"]; shot=item["shot"];a.launch(shot,card=c);a.partner.reset(seed=c["seed"])
                for _ in range(190):
                    _,_,done,cap,_=a.exchange(u,mpc.FAR_COMMAND)
                    if done or cap or 1 in a.crossings:
                        break
                o=delivery(a)
                target=a.point_placements()["requested"]
                if target and target[0]["target"]!=list(draw(c,0,"random",None)[0]):
                    raise ValueError("Actual randomized opponent draw changed")
                dim=None
                if o and o.get("receive_line"):
                    z=o["receive_line"]
                    dim=old_look.central.deep(np.r_[z["xyz"],z["velocity"]]).tolist()
                report["rows"].append(dict(item,physical=None if o is None else buzz(o),true_deep=dim))
                if len(report["rows"])%12==0:
                    _write(SCOUT,report)
                    if all(len(v)>=4 for v in strata(report["rows"])):
                        break
        finally:a.close()
    center,low,rout=strata(report["rows"])
    roster=[]
    for name,rows,n in (("central-low",center,4),("wing-low",low,4),("routine",rout,4)):
        if name!="routine":
            rows=sorted(rows,key=lambda r:(abs(r["true_deep"][1]-1.003),abs(r["true_deep"][0])))
        else:
            rows=sorted(rows,key=lambda r:abs(r["true_deep"][1]-1.086))
        roster.extend(dict(r,role=name) for r in rows[:n])
    report.update(complete=True,optical_roster=roster,scope="First completed 12-card scout-boundary geometry quota before rendering or receiving. No optical outcome selected hold rows.")
    _write(SCOUT,report)
    print("TT93 optically held before actuation",[(r["role"],r["card"]["id"],r["true_deep"][:2]) for r in roster],flush=True)


def ray(cam,s):
    u=cam.ray(s.center);return (cam.eye,u/np.linalg.norm(u))


def triangulate(rays,weights=None):
    """Geometric multi-view position using synchronized RGB centroids only."""
    if len(rays)<2:return None
    if weights is None:weights=np.ones(len(rays))
    a=np.zeros((3,3));b=np.zeros(3)
    for (eye,unit),w in zip(rays,weights,strict=True):
        M=np.eye(3)-np.outer(unit,unit)
        a+=w*M;b+=w*(M@eye)
    if np.min(np.linalg.eigvalsh(a))<.025:
        return None
    p=np.linalg.solve(a,b)
    if any((p-eye)@u<.02 for eye,u in rays):
        return None
    err=max(np.linalg.norm(np.cross(p-eye,u)) for eye,u in rays)
    return (p,float(err)) if err<.022 else None


def measure(rig,frames,q):
    cams={**rig.fixed,"wrist":rig.wrist.at(q)}
    sights={s:native_ball_pixels(f) for s,f in frames.items()}
    hits={s:ray(cams[s],v) for s,v in sights.items() if v is not None and not v.clipped}
    stereo=triangulate([hits[k] for k in ("left","right") if k in hits])
    alternate=triangulate([hits[k] for k in ("left","wrist") if k in hits])
    all_three=triangulate([hits[k] for k in ("left","right","wrist") if k in hits],
                         [1.,1.,1.7] if all(k in hits for k in ("left","right","wrist")) else None)
    return cams,sights,hits,stereo,alternate,all_three


def correction(cam,sighting,position):
    """Use a close wrist angle to refine transverse position, retaining range."""
    if sighting is None or sighting.clipped or sighting.area<7:
        return None
    at=cam.project(position)
    if at[2]<.035 or not np.isfinite(at).all() or np.linalg.norm(at[:2]-sighting.center)>13.:
        return None
    eye,u=ray(cam,sighting)
    delta=eye+u*((position-eye)@u)-position
    return .62*delta if np.linalg.norm(delta)<.026 else None


def observed(a,cams,sights,t,results):
    pr=a.ball.copy();v=a.velocity.copy()
    cc=a.data.geom("learner_racket")
    axes=cc.xmat.reshape(3,3)
    p=pr-cc.xpos
    cam=cams.get("wrist"); sc=sights.get("wrist"); pos=cam.project(pr)
    expected_radius=cam.focal*.020/max(.003,pos[2])
    optical=bool(pos[2]>.012 and sc is not None and
                 np.linalg.norm(pos[:2]-sc.center)<max(3.,min(12.,.52*expected_radius)))
    values=[]
    for answer in results:
        if answer is None:
            values.append(None);continue
        got,error=answer
        u=axes.T@(got-pr)
        values.append(dict(p=got.tolist(),error_m=float(np.linalg.norm(u)),
                           tangential_m=float(np.linalg.norm(u[:2])),separation_m=error))
    return dict(tick=t,time=float(a.data.time),after=bool(a._received_bounce and v[1]<-.8),
            ball=pr.tolist(),velocity=v.tolist(),patch=(axes.T@p).tolist(),
            wrist_match=optical,wrist_clipped=None if sc is None else sc.clipped,
            raw_pixel=None if sc is None else sc.center.tolist(),prediction=pos.tolist(),
            bearing_m=old_look.angular_error(cam,sc.center,pr) if optical else None,
            stereo=values[0],left_wrist=values[1],all_three=values[2])


def act_setup():
    ready=np.asarray(_read(PLAN)["home"])
    ranking=season.fresh.shot.Coverage(_read(season.fresh.RISK))
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    return ready,raw,ranking,_read(season.CARRY)


def actor_for(raw,rank,cp,ids):
    with patch.object(mpc,"Fragments",lambda f,e:Carried(f,e,practice=[course.PRACTICE],held=ids)):
        actor=season.Player(mpc.BANK,raw,coverage=rank,variant="elastic",budget=28)
    course.resume(actor,season.CARRY.with_suffix("").with_suffix(".npz"),cp)
    return actor


def optical(a,res,actor,ready,entry):
    cam,rig,stride,lower,upper=res
    c=entry["card"];u_ref=ready.copy();b=a.model.geom("learner_racket").id
    a.launch(entry["shot"],card=c);a.partner.reset(seed=c["seed"])
    eye=mpc.Vision(rig,a.control_dt,raw=True);eye.eye=mpc.LearnedOptics(eye.eye,course.OPTIC)
    actor.begin();records=[];strokes=[];samples={}; crossing=None; gate=False
    assessment=mount.Clear(a)
    for tick in range(230):
        q=a.joints.copy()
        f=cam.capture() if tick%2==0 else None
        x=mpc.observed(eye,tick,f,q)
        close=x is not None and x[4]<-.90 and -1.7<x[1]<-.34 and x[2]>.77
        if close:
            gate=True
            w=a.render_wrist()
            image={"wrist":w} if f is None else {**f,"wrist":w}
            cams,sights,hits,*data=measure(rig,image,q)
            r=observed(a,cams,sights,tick,data); records.append(r)
            d=correction(cams["wrist"],sights.get("wrist"),x[:3])
            if d is not None:
                axes=a.data.geom_xmat[b].reshape(3,3)
                e=axes.T@(x[:3]-a.ball)
                repaired=axes.T@(x[:3]+d-a.ball)
                r["online_tangential_m"]=[float(np.linalg.norm(e[:2])),float(np.linalg.norm(repaired[:2]))]
            r["module_clearance"]=assessment.at(q)
            rr,gg,bb=np.moveaxis(w.astype(np.int16),-1,0)
            r["red_pixels"]=int(np.sum((rr>80)&(rr-gg>47)&(rr-bb>36)&(gg-bb<16)&(gg<.56*rr)))
            if r["wrist_match"] and not r["wrist_clipped"]:
                samples[tick]=w.copy()
        if a.ball[1]<-1.53 and a.velocity[1]<-.8 and crossing is None:
            crossing=float(a.data.time-DT*.5)
        g=ready.copy()
        if x is not None and (x[4]<-.9 or gate and x[1]<.50):
            target,confidence=actor.raw.get(x,q)
            if target is not None:g=ready+confidence*(target-ready)
        want=actor.step(tick,eye.eye,x,q,u_ref,g)
        u=np.clip((np.clip(want,lower,upper)-u_ref)/stride,-.8,.8); u_ref=np.clip(u_ref+u*stride,lower,upper)
        _,_,done,stop,_=a.exchange(u,mpc.FAR_COMMAND)
        strokes.append(u.tolist())
        actual=delivery(a)
        if done or stop or actual and actual["near_contact"] or crossing is not None and a.data.time>crossing+.06:
            break
    o=delivery(a)
    if o is None or buzz(o)!=entry["physical"]:
        raise ValueError("Optical arm did not receive the same sampled native goal and physical incoming")
    clock=float(o["near_contact"]["time"]) if o["near_contact"] else crossing
    for r in records:
        r["lead_s"]=None if clock is None else float(clock-r["time"])
    usable=[r for r in records if r["lead_s"] is not None and 4*DT<=r["lead_s"]<=.15 and r["after"]
            and r["wrist_match"] and not r["wrist_clipped"] and r["red_pixels"]>=30]
    if usable:
        from PIL import Image
        r=usable[0]; target=ROOT/"frames";target.mkdir(exist_ok=True)
        Image.fromarray(samples[r["tick"]]).save(target/f"rgb-{entry['role']}-{c['id']}-tick{r['tick']}.png")
    return dict(records=records,clock=clock,offer=o,actions=strokes,events=a.events,usable=len(usable))


def evaluate():
    p,m=guaranteed(); qualified=_read(SCOUT)
    if not qualified["complete"] or qualified["model_sha256"]!=_hash(mount.MODEL):
        raise ValueError("Unsealed optical roster")
    home,raw,rank,cp=act_setup(); roster=qualified["optical_roster"]
    result=dict(scout_sha256=_hash(SCOUT),model_sha256=_hash(mount.MODEL),rows=[],complete=False)
    if HOLD.exists():
        r=_read(HOLD)
        if r["scout_sha256"]!=result["scout_sha256"] or r["model_sha256"]!=result["model_sha256"]:
            raise ValueError("Resume requires same receipts")
        result=r
    with mount.builder(m),mpc.centered(home),school.enabled():
        a=mpc.arena()
        res=mpc.resources(a)
        actor=actor_for(raw,rank,cp,{r["card"]["id"] for r in roster})
        try:
            for r in roster[len(result["rows"]):]:
                observed=optical(a,res,actor,home,r)
                report=dict(case=r,**observed);result["rows"].append(report);_write(HOLD,result)
                print(r["role"],r["card"]["id"],observed["usable"],len(observed["records"]),flush=True)
        finally:
            actor.close();res[0].close();a.close()
    result["complete"]=True;_write(HOLD,result)


def summary():
    d=_read(HOLD)
    if not d["complete"]:raise ValueError("Finish held captures")
    notes=[]
    all_errors=[]
    holds=[]
    unsafe=0
    for step in d["rows"]:
        r=[r for r in step["records"] if r["lead_s"] is not None and 4*DT<=r["lead_s"]<=.15 and r["after"]
           and r["wrist_match"] and not r["wrist_clipped"] and r["red_pixels"]>=30]
        vals=[(v["stereo"]["error_m"],v["left_wrist"]["error_m"],v["all_three"]["error_m"],
               v["stereo"]["tangential_m"],v["all_three"]["tangential_m"])
                for v in r if all(v[k] is not None for k in ("stereo","left_wrist","all_three"))]
        all_errors.extend(vals)
        phase=[v["online_tangential_m"] for v in r if "online_tangential_m" in v]
        holds.extend(phase)
        unsafe+=sum(bool(v["module_clearance"]) for v in step["records"])
        notes.append(dict(id=step["case"]["card"]["id"],role=step["case"]["role"],useful=len(r)>=2,
                          usable=len(r),matched_triangulations=len(vals),
                          medians_mm=None if not vals else (1000*np.median(vals,axis=0)).tolist()))
    med=None if not all_errors else (1000*np.median(all_errors,axis=0)).tolist()
    phase_med=None if not holds else (1000*np.median(holds,axis=0)).tolist()
    hard=sum(r["role"]!="routine" and r["useful"] for r in notes)
    signal=bool(phase_med and phase_med[1]<.9*phase_med[0] and len(holds)>=12 and hard>=3 and not unsafe)
    _write(REPORT,dict(physical_capture_sha256=_hash(HOLD),cases=notes,median_errors_mm=med,good_hard=hard,
            median_fields=("left+right","left+wrist","all three","left+right tangential","all three tangential"),
            paired_frames=len(all_errors),fusion_paired_frames=len(holds),causal_bearing_tangential_mm=phase_med,
            module_penetration_frames=unsafe,admit_prospective_fusion=signal))
    print("optics",len(notes),hard,len(all_errors),med,phase_med,"admit",signal,flush=True)


if __name__=="__main__":
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument("mode",choices=("plan","scout","evaluate","summary"))
    {"plan":plan,"scout":scout,"evaluate":evaluate,"summary":summary}[q.parse_args().mode]()
