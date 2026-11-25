"""Learn bounded early joint moves by native outcome trials on identical optical offers."""

import argparse
from pathlib import Path
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt43_temporal_servo as servo
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_carry_joints import Carried
from .tt44_paired_credit import PLAN, State, Teacher, prefix, sources, suffix, verify_restore

ROOT=Path("artifacts/tt44_continuous_credit")
PAIRED=Path("artifacts/tt44_paired_credit/training.json.gz")
# Signed early progress, then five joint-reference offsets.
ENVELOPE=np.array([10.,.43,.57,.62,.51,.59])
SEEDS=np.array([[.83,0.,0.,0.,0.,0.],
                [.27,-.16,.75,.68,-.53,-.12],
                [-.23,.28,-.57,.70,.51,.35],
                [.50,-.29,.58,-.56,-.43,-.50]])
ARM_CUE=(-.53,1.32)


def sensed_look(x,ticks):
    h=np.clip(ticks*mpc.DT,-.06,.18)
    if h>0.:
        h=min(h,(x[1]+.56)/max(-x[4],1.15))
    look=x.copy()
    look[:3]+=h*x[3:6]
    look[2]-=4.905*h*h
    look[5]-=9.81*h
    if look[2]<.79:
        return x
    return look


class Continuous(Teacher):
    def __init__(self,*args,parameter=None,outbound=False,chooser=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.parameter=np.zeros(6) if parameter is None else np.asarray(parameter)
        self.outbound=outbound
        self.chooser=chooser
        self.active=None

    def begin(self):
        super().begin()
        self.active=None
        self.outbound_observed=False
        self.commands_modified=0

    def get_base(self,tick,x,q,reference,home):
        base=super().get_base(tick,x,q,reference,home)
        if x is None:
            return base
        if x[4]>.9:
            if self.outbound and -.04<x[1]<1.10:
                self.outbound_observed=True
                out=base.copy()
                out[1:5]+=.55*self.parameter[2:6]*ENVELOPE[2:6]
                self.diag["outbound_prep_ticks"]+=1
                return out
            self.active=None
        if x[4]>=-.90 or not ARM_CUE[0]<x[1]<ARM_CUE[1]:
            return base
        if self.active is None:
            if self.chooser is None:
                self.active=self.parameter.copy()
            elif self.context is not None:
                self.active=self.chooser.get(self.context)
            else:
                return base
        k=self.active
        if not np.any(k):
            return base
        look=sensed_look(x,k[0]*ENVELOPE[0])
        move,conf=self.raw.get(look,q)
        lead=base.copy() if move is None else home+conf*(move-home)
        frame=np.clip((x[1]+.53)/.31,0.,1.)*np.clip((1.32-x[1])/.47,0.,1.)
        # Encoder position and its error relative to prior references guard
        # the offsets against forcing a motor already far behind.
        offset=k[1:]*ENVELOPE[1:]
        lag=reference[:5]-q[:5]
        against=offset*lag>0.
        attenuate=np.where(against,1.-.38*np.clip(abs(lag)/.28,0.,1.),1.)
        delta=np.clip(lead[:5]-base[:5]+offset*attenuate,-.68,.68)
        out=base.copy()
        out[:5]+=frame*delta
        self.diag["continuously_corrected_ticks"]+=int(np.max(abs(frame*delta))>1e-6)
        if self.approaches:
            self.approaches[-1]["parameters"]=k.tolist()
        return out


def measure_effect(a,old,case,run):
    # Evaluation only. The actor never receives true contact or departure.
    if run["landed"]:
        return -4.0,dict(legal=True)
    touch=run["first_contact"]
    e=next((e for e in a.events if touch and e.get("collider")=="learner_racket"
            and e["time"]>touch["time"] and e["time"]-touch["time"]<.065),None)
    if e is not None:
        p,v=np.asarray(e["ball"]),np.asarray(e["velocity"])
        if v[1]>.12:
            h=max(0.,-p[1]/max(v[1]*.9,.12))
            net=p[2]+h*v[2]-4.905*h*h
            landing=(v[2]+np.sqrt(max(0.,v[2]**2+19.62*(p[2]-.78))))/9.81
            destination=p[:2]+v[:2]*landing*.90
            loss=(.65+3.9*max(0.,.96-net)+1.5*abs(destination[1]-.63)
                  +2.1*max(0.,abs(destination[0])-.62))
            effect=dict(departure_xyz=e["ball"],departure_velocity=e["velocity"],net_proxy_m=float(net),legal=False)
            return float(min(loss,6.5)),effect
        return 6.6,dict(departure_xyz=e["ball"],departure_velocity=e["velocity"],legal=False)
    # Soft ranking distinguishes a native touch without separation from
    # no touch. Encoder FK is offline assistance for a second-round draw.
    if touch:
        return 7.0,dict(legal=False)
    qkin=mpc.BladeIK(a)
    gaps=[float(np.linalg.norm(qkin.pose(np.asarray(f[7:12]))[0]-np.asarray(f[1:4])))
          for f in run["sensory"] if -.9>f[2]>-1.74 and f[5]<-.9]
    gap=min(gaps,default=.6)
    return 8.5+min(gap,1.)*4.,dict(closest_optical_fk_m=gap,legal=False)


def reset_setup(a,c,old,image,boot,outbound):
    current=c.copy()
    if outbound:
        offered=old["placements"]["delivered"][c["leg"]]
        earlier=[e for e in old["events"] if e.get("collider")=="learner_racket" and
                 e["time"]<offered["far_contact"]["time"]-.16]
        time=earlier[-1]["time"]+.067 if earlier else .27
        current["tick"]=int(np.ceil(time/mpc.DT-1e-7))
    readings=prefix(a,c["card"],old,current["tick"],image,a._arm_ids[1][2])
    frozen=State(a)
    verify_restore(a,frozen,old,min(c["tick"]+85,len(old["actions"])),current)
    return current,readings,frozen


def collect(plan,paired,output,limit,batch,refine,outbound,ids=None):
    ps=_read(plan)
    base=_read(paired)
    if base["plan_sha256"]!=_hash(plan) or not base.get("complete"):
        raise ValueError("Need the completed matched baseline")
    old_series=sources()
    roster=[c for c in ps["train"] if c["stratum"]!="easy" and (ids is None or c["id"] in ids)][:limit]
    if any(c["id"] in {r["id"] for r in ps["held"]} for c in roster) or not roster:
        raise ValueError("Training overlap or empty trial")
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    boot=np.asarray(_read(mpc.BANK.with_suffix(".json"))["ready"])
    motor=servo.reload(servo.ROOT/"prior.npz")
    report=dict(schema="tt44-continuous-causal-practice-1",pairs_sha256=_hash(paired),plan_sha256=_hash(plan),
                batch=batch,refine=refine,outbound=outbound,bounds=ENVELOPE.tolist(),roster=roster,rows=[])
    if output.exists():
        before=_read(output)
        if any(before.get(k)!=v for k,v in report.items() if k!="rows"):
            raise ValueError("Output is from a different practice condition")
        report=before
    with mpc.centered(boot):
        a=mpc.arena()
        camera,rig,stride,lo,hi=mpc.resources(a)
        with patch.object(mpc,"Fragments",lambda b,e:Carried(b,e,practice=[course.PRACTICE],
                                                             held={c["id"] for c in ps["held"]})):
            actor=Continuous(mpc.BANK,raw,variant="elastic",budget=28,policy="stock",
                             motor=motor,outbound=outbound)
        try:
            for c in roster[len(report["rows"]):]:
                _,source=old_series[c["source"]]
                old=source["rows"][c["index"]]
                case,memory,saved=reset_setup(a,c,old,camera,boot,outbound)
                rng=np.random.default_rng([444582,c["id"],int(outbound)])
                initial=SEEDS[:batch] if batch<=len(SEEDS) else np.vstack((SEEDS,rng.uniform(-.75,.75,(batch-len(SEEDS),6))))
                runs=[]
                def drive(z):
                    saved.restore(a)
                    actor.parameter=np.array(z)
                    result=suffix(a,camera,rig,stride,lo,hi,actor,memory,case,old,boot)
                    loss,native=measure_effect(a,old,case,result)
                    result.update(parameters=z.tolist(),physical_loss=loss,physical_effect=native)
                    print(c["id"],len(runs),result["landed"],result["touched"],round(loss,3),flush=True)
                    runs.append(result)
                begun=process_time()
                if outbound:
                    drive(np.zeros(6))
                    ref=runs[0]
                else:
                    ref=next(r["variants"]["stock"] for r in base["rows"] if r["id"]==c["id"])
                for z in initial:
                    drive(z)
                    if runs[-1]["landed"] and not ref["landed"]:
                        break
                if not any(r["landed"] and np.any(r["parameters"]) for r in runs):
                    top=min((r for r in runs if np.any(r["parameters"])),key=lambda r:r["physical_loss"])
                    # Actual contact or very close body-supported attempt
                    # earns further local draws. No hindsight by point ID.
                    if top["physical_loss"]<9.35:
                        for _ in range(refine):
                            candidate=np.clip(np.asarray(top["parameters"])+rng.normal(0.,.31,6),-1.,1.)
                            drive(candidate)
                            if runs[-1]["landed"]:
                                break
                report["rows"].append(dict(case,stock_landed=ref["landed"],stock_touched=ref["touched"],
                    stock_context=None if not ref["decisions"] else ref["decisions"][0]["context"],
                    trials=runs,modified_landed=int(sum(r["landed"] and np.any(r["parameters"]) for r in runs)),
                    native_restore_verified=True,cpu_s=process_time()-begun))
                _write(output,report)
        finally:
            actor.close();camera.close();a.close()
    report["complete"]=True
    _write(output,report)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--paired",type=Path,default=PAIRED)
    p.add_argument("--output",type=Path,default=ROOT/"training.json.gz")
    p.add_argument("--limit",type=int,default=8)
    p.add_argument("--batch",type=int,default=4)
    p.add_argument("--refine",type=int,default=2)
    p.add_argument("--outbound",action="store_true")
    p.add_argument("--ids",type=int,nargs="+")
    a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    collect(a.plan,a.paired,a.output,a.limit,a.batch,a.refine,a.outbound,a.ids)
