"""Collect paired causal preparation credit at exact saved physical receiving states."""

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import is_dataclass
from pathlib import Path
from time import process_time
from unittest.mock import patch

import mujoco
import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt43_temporal_servo as servo
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent import draw
from .tt39_carry_joints import Carried
from .tt39_opportunity_map import projected_height
from .tt40_long_low import same_trace
from .tt44_preview import eligible, preview

ROOT=Path("artifacts/tt44_paired_credit")
SRC=course.ROOT/"learning-24.json.gz"
OLDER=Path("artifacts/tt37/mix-rollin.json.gz")
RESERVED=Path("artifacts/tt44_combined/fresh.json")
PLAN=ROOT/"plan.json"
TEACHERS=("stock","temporal","preview4")
# Prespecified across archived terminal losses and ordinary legal controls.
# One offer per release, many openings to keep replay cost reasonable.
CASES=(("low",14502,0),("miss",9519,0),("easy",4636,0),
       ("low",10326,1),("miss",4450,0),("easy",3178,0),
       ("low",11365,2),("miss",9080,1),("easy",13896,0),
       ("low",4239,0),("miss",4283,2),("easy",5349,1))
CONTROL=course.ROOT/"after12.json.gz"
ASSESSMENT=course.ROOT/"direct-course.json"


def sources():
    return {p.name:(p,_read(p)) for p in (SRC,OLDER)}


def freeze(out):
    previous=sources()
    reserved={c["id"] for cards in _read(RESERVED).values() if isinstance(cards,list)
                           for c in cards if isinstance(c,dict) and "id" in c}
    rows=[]
    for stratum,ident,leg in CASES:
        options=[(file,d,i) for file,d in previous.values() for i,c in enumerate(d["cards"][:len(d["rows"])])
                 if c["id"]==ident]
        p,d,i=options[0]
        card,row=d["cards"][i],d["rows"][i]
        o=row["placements"]["delivered"][leg]
        h=projected_height(o["receive_line"])
        if (ident in reserved or card.get("profile","reach")!="reach" or card.get("opening")!="random"
                or not o["near_bounce"] or stratum=="easy" and (not o["near_landed"] or h<1.045)
                or stratum=="miss" and (o["near_landed"] or h<.99)
                or stratum=="low" and (o["near_landed"] or not .95<h<.99)):
            raise ValueError(f"Invalid predeclared training stratum {stratum} {ident} {h}")
        far=next(e for e in row["events"] if e.get("collider")=="rival_racket"
                 and o["far_contact"]["time"]<=e["time"]<o["near_bounce"]["time"])
        tick=int(np.ceil(far["time"]/mpc.DT-1e-7))
        rows.append(dict(id=ident,leg=leg,source=p.name,index=i,stratum=stratum,tick=tick,
                         coach_landed=o["near_landed"],projected_height_m=float(h),
                         incoming_separation_s=float(far["time"]),card=card))
    held=[]
    for i,c in enumerate(_read(ASSESSMENT)["assessment"]):
        if c["id"] not in reserved|{r["id"] for r in rows} and 2<=i<6:
            held.append(c)
    out.parent.mkdir(parents=True,exist_ok=True)
    _write(out,dict(schema="tt44-paired-credit-plan-1",train=rows,held=held,teachers=TEACHERS,
                    sources={p.name:_hash(p) for p,d in previous.values()},reserved_sha256=_hash(RESERVED),
                    stage="Same native state immediately after far separation. Only ordinary RGB/joint preparation between there and the unchanged late planner."))


def copyable(v):
    return isinstance(v,(str,bool,int,float,type(None),tuple,list,set,dict,np.ndarray,np.generic,
                         np.random.Generator)) or is_dataclass(v)


class State:
    """Copy the native solver workspace and ordinary Python referee/opponent state."""
    def __init__(self,a):
        self.native=mujoco.MjData(a.model)
        mujoco.mj_copyData(self.native,a.model,a.data)
        self.dynamic=self._save(a)
        self.partner=self._save(a.partner)
        self.force=a.model.actuator_forcerange.copy()

    @staticmethod
    def _save(o):
        # Object handles such as renderers/compiled flight models stay in the
        # same arena. All numerical/list/lifecycle state is restored per fork.
        return {k:deepcopy(v) for k,v in vars(o).items() if k!="_cad_reader" and copyable(v)}

    def restore(self,a):
        mujoco.mj_copyData(a.data,a.model,self.native)
        a.model.actuator_forcerange[:]=self.force
        for target,d in ((a,self.dynamic),(a.partner,self.partner)):
            target.__dict__.update(deepcopy(d))


def features(x,q,ref,first):
    """One causal choice at a common decision row before any teacher can differ."""
    return np.r_[x[:3]/[.27,1.2,.24], x[3:6]/[.85,1.75,1.3],
                 (x-first)/[.17,.35,.11,.48,.84,1.],
                 q[:5]/[.6,1.8,1.8,1.4,1.2], (ref[:5]-q[:5])/.16]


class Teacher(mpc.MPC):
    def __init__(self,*args,policy,motor,selector=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.policy=policy
        self.motor=motor
        self.selector=selector

    def begin(self):
        super().begin()
        self.sensed=False
        self.inbound=deque(maxlen=7)
        self.context=None
        self.choice=None
        self.approaches=[]

    def get_base(self,tick,x,q,reference,home):
        if x is None or x[4]>=-.9:
            self.inbound.clear()
            if x is not None and x[4]>.9:
                self.choice=None
                self.context=None
        else:
            self.inbound.append(np.r_[x,q[:5]])
        if x is not None and x[4]<-.9 and x[1]<-.90:
            self.sensed=True
        base=home
        if x is not None and (x[4]<-.9 or (self.sensed and x[1]<.50)):
            change,confidence=self.raw.get(x,q)
            if change is not None:
                base=home+confidence*(change-home)
        # Make the choice at the same eligible public row for all alternatives.
        if self.context is None and eligible(x) and len(self.inbound)>=3:
            self.context=features(x,q,reference,self.inbound[-3][:6])
            self.choice=self.selector.choose(self.context) if self.selector is not None else self.policy
            self.approaches.append(dict(tick=tick,mode=self.choice,context=self.context.tolist()))
        if self.choice=="temporal" and len(self.inbound)==7 and -.57<x[1]<1.35 and x[4]<-.9:
            estimate=q[:5]+self.motor.predict(servo.feature(self.inbound))[0]
            base=base.copy()
            base[:5]+=.65*np.clip(estimate-base[:5],-.25,.25)
            self.diag["preparation_ticks"]+=1
        elif self.choice=="preview4" and eligible(x):
            target,conf=self.raw.get(preview(x,4),q)
            if target is not None:
                desired=home+conf*(target-home)
                adjust=np.clip(desired-base,-np.array([.42,.32,.33,.36,.43,0.]),
                              np.array([.42,.32,.33,.36,.43,0.]))
                self.diag["preparation_ticks"]+=int(np.max(abs(adjust))>1e-6)
                base=base+adjust
        return base


def prefix(a,c,row,tick,image,axes):
    a.launch(row["shot"],card=c)
    a.partner.reset(seed=c["seed"])
    readings=[]
    for k,u in enumerate(row["actions"][:tick]):
        # Replay every body reference, but warm the RGB fit only on the
        # recent outgoing segment. It sees no future exposures.
        frame=image.capture() if k>=tick-22 and k%2==0 else None
        readings.append((k,frame,a.joints.copy(),a.data.ctrl[axes].copy()))
        a.exchange(u,mpc.FAR_COMMAND)
    expected=[e for e in row["events"] if e["time"]<=a.data.time+1e-8]
    if digest_json(a.events)!=digest_json(expected):
        raise ValueError("Physical tape diverged before the fork")
    return readings


def warm(a,rig,actor,readings):
    eye=mpc.Vision(rig,a.control_dt,raw=True)
    eye.eye=mpc.LearnedOptics(eye.eye,course.OPTIC)
    actor.begin()
    for t,image,q,ref in readings:
        x=mpc.observed(eye,t,image,q)
        actor.engine.observe(t,eye.eye,q,ref.copy())
    return eye


def suffix(a,image,rig,stride,low,high,actor,readings,case,source,boot):
    eye=warm(a,rig,actor,readings)
    leg=case["leg"]; tick=case["tick"]
    axes=a._arm_ids[1][2]
    target_bounce=source["placements"]["delivered"][leg]["near_bounce"]["time"]
    controls=[]; sensory=[]; landed=False
    for j in range(tick,tick+170):
        q=a.joints
        ref=a.data.ctrl[axes].copy()
        x=mpc.observed(eye,j,image.capture() if j%2==0 else None,q)
        base=actor.get_base(j,x,q,ref,boot)
        want=actor.step(j,eye.eye,x,q,ref,base)
        u=np.clip((np.clip(want,low,high)-ref)/stride,-.8,.8)
        sent=np.clip(ref+u*stride,low,high)
        if x is not None and x[4]<-.9:
            sensory.append([j,*x,*q,*ref,*sent,*u])
        controls.append(u.tolist())
        _,_,finished,capped,info=a.exchange(u,mpc.FAR_COMMAND)
        if any(e["kind"]=="legal_return" and e.get("side")==1 and e["time"]>target_bounce
               for e in info["events"]):
            landed=True
        if landed or finished or capped:
            break
    placements=a.point_placements()
    delivered=placements["delivered"]
    if leg>=len(delivered):
        raise ValueError("Expected the matched delivered ball")
    offer=delivered[leg]
    expected=source["placements"]["delivered"][leg]
    if (not all(same_trace(offer[k],expected[k]) for k in ("far_contact","near_bounce"))
            or bool(offer["near_landed"])!=landed):
        raise ValueError("Intervention did not retain the identical physically offered ball")
    if [v["target"] for v in placements["requested"]]!=[list(draw(case["card"],i,"random",None)[0])
                                                        for i in range(len(placements["requested"]))]:
        raise ValueError("Opponent did not use exact assigned random draws")
    return dict(landed=landed,touched=offer["near_contact"] is not None,
                first_contact=offer["near_contact"],point_fault=next((e.get("reason") for e in a.events[::-1]
                     if e["kind"]=="point"),None),decisions=actor.approaches,
                diagnostics=dict(actor.diag),controls=controls,sensory=sensory,
                events_sha256=digest_json(a.events),exact_offer=True,truncated=not (landed or finished))


def verify_restore(a,saved,old,stop,case):
    saved.restore(a)
    for u in old["actions"][case["tick"]:stop]:
        a.exchange(u,mpc.FAR_COMMAND)
    expected=[e for e in old["events"] if e["time"]<=a.data.time+1e-8]
    if digest_json(a.events)!=digest_json(expected):
        raise ValueError("Saved native state does not reproduce archived suffix")


def collect(plan,output,limit):
    frozen=_read(plan)
    previous=sources()
    train=frozen["train"][:limit]
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    boot=np.array(_read(mpc.BANK.with_suffix(".json"))["ready"])
    motor=servo.reload(servo.ROOT/"prior.npz")
    receipt=dict(schema="tt44-paired-credit-training-1",plan_sha256=_hash(plan),teachers=TEACHERS,
                 temporal_sha256=_hash(servo.ROOT/"prior.npz"),optic_sha256=_hash(course.OPTIC),
                 input_fields=["tick","stereo_p_v_6","encoders_6","prior_ref_6","sent_ref_6","motor_rates_6"],
                 rows=[])
    with mpc.centered(boot):
        a=mpc.arena()
        camera,rig,stride,low,high=mpc.resources(a)
        held={r["id"] for r in frozen["held"]}
        with patch.object(mpc,"Fragments",lambda path,e:Carried(path,e,practice=[course.PRACTICE],held=held)):
            policies={name:Teacher(mpc.BANK,raw,variant="elastic",budget=28,policy=name,motor=motor)
                      for name in TEACHERS}
        try:
            for case in train:
                p,d=previous[case["source"]]
                if d["ready"]!=boot.tolist() or _hash(p)!=frozen["sources"][p.name]:
                    raise RuntimeError("Archive ABI or physical boot changed")
                old=d["rows"][case["index"]]
                start=process_time()
                readings=prefix(a,case["card"],old,case["tick"],camera,a._arm_ids[1][2])
                saved=State(a)
                verify_restore(a,saved,old,min(case["tick"]+85,len(old["actions"])),case)
                outcomes={}
                for name,actor in policies.items():
                    saved.restore(a)
                    outcomes[name]=suffix(a,camera,rig,stride,low,high,actor,readings,case,old,boot)
                anchors=[v["decisions"][0] if v["decisions"] else None for v in outcomes.values()]
                if any((e is None)!=(anchors[0] is None) for e in anchors[1:]):
                    raise ValueError("Unexpected predecision visibility change")
                if anchors[0] and any(e["context"]!=anchors[0]["context"] or e["tick"]!=anchors[0]["tick"] for e in anchors[1:]):
                    raise ValueError("The teachers changed action before the first public decision")
                d=dict(case,variants=outcomes,common_context=None if not anchors[0] else anchors[0]["context"],
                       native_restore_verified=True,cpu_s=process_time()-start)
                receipt["rows"].append(d)
                _write(output,receipt)
                print(case["id"],case["leg"],case["stratum"],[(k,v["landed"],v["touched"],
                      v["diagnostics"].get("preparation_ticks",0)) for k,v in outcomes.items()],flush=True)
        finally:
            for actor in policies.values(): actor.close()
            camera.close();a.close()
    receipt["complete"]=True
    _write(output,receipt)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("plan","collect"))
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--output",type=Path,default=ROOT/"training.json.gz")
    p.add_argument("--limit",type=int,default=len(CASES))
    a=p.parse_args()
    freeze(a.plan) if a.phase=="plan" else collect(a.plan,a.output,a.limit)
