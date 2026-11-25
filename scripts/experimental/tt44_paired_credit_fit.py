"""Freeze a paired-return uplift selector and check it in complete disjoint RGB games."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt43_temporal_servo as servo
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_opponent import draw
from .tt39_carry_joints import Carried
from .tt40_long_low import same_trace
from .tt44_paired_credit import PLAN, ROOT, TEACHERS

from .tt44_paired_credit import Teacher

STOCK=course.ROOT/"no-more-practice.json.gz"
SCALE=np.r_[[1.]*6,[.8]*6,[.75]*5,[.65]*5]
MIN_GAIN=.065
MIN_HELP=1.10
PRIOR=1.65


class Selector:
    def __init__(self,context,uplift):
        self.context=np.asarray(context).reshape((-1,len(SCALE)))
        self.uplift=np.asarray(uplift).reshape((-1,len(TEACHERS)-1))

    def choose(self,z):
        d=np.mean(((self.context-z)/SCALE)**2,axis=1)
        w=np.exp(-2.2*d)
        score=(w@self.uplift)/(w.sum()+PRIOR)
        support=w@(self.uplift>0)
        j=int(np.argmax(score))
        if score[j]<MIN_GAIN or support[j]<MIN_HELP:
            return "stock"
        return TEACHERS[j+1]


def outcome(row):
    return np.array([int(row["variants"][s]["landed"]) for s in TEACHERS])


def movement(row, policy):
    a,b=(row["variants"][k] for k in ("stock",policy))
    raw={int(f[0]):np.asarray(f) for f in a["sensory"] if -.57<f[2]<1.35}
    alt={int(f[0]):np.asarray(f) for f in b["sensory"] if -.57<f[2]<1.35}
    comparable=set(raw)&set(alt)
    shifts=[np.max(abs(raw[t][7:12]-alt[t][7:12])) for t in comparable]
    first=next((i for i,(u,v) in enumerate(zip(a["controls"],b["controls"])) if u!=v),None)
    return dict(first_different_control=None if first is None else row["tick"]+first,
                common_early_samples=len(comparable), maximum_early_encoder_shift_rad=float(max(shifts,default=0.)))


def fit(path,plan,output):
    d=_read(path); schedule=_read(plan)
    if not d.get("complete") or d["plan_sha256"]!=_hash(plan) or len(d["rows"])!=len(schedule["train"]):
        raise ValueError("Freeze only fully paired preassigned practice")
    rows=d["rows"]
    censored=[r for r in rows if r["common_context"] is None or any(v["truncated"] for v in r["variants"].values())]
    fitrows=[r for r in rows if r not in censored]
    x=np.array([r["common_context"] for r in fitrows])
    y=np.array([outcome(r) for r in fitrows])
    u=y[:,1:]-y[:,0:1]
    variant=np.array([outcome(r) for r in rows])
    mdl=Selector(x,u)
    diagnostic=[]
    for i,r in enumerate(fitrows):
        keep=np.arange(len(y))!=i
        mode=Selector(x[keep],u[keep]).choose(x[i])
        diagnostic.append(dict(id=r["id"],stock=bool(y[i,0]),action=mode,
                               selected_landed=bool(y[i,TEACHERS.index(mode)]),
                               best_counterfactual=bool(np.max(y[i]))))
    by={}
    for s in {r["stratum"] for r in rows}:
        subset=np.array([outcome(r) for r in rows if r["stratum"]==s])
        by[s]=dict(offers=len(subset),teacher_legal=dict(zip(TEACHERS,map(int,subset.sum(0)))),
                   any_teacher_legal=int(np.max(subset,axis=1).sum()))
    _write(output,dict(schema="tt44-paired-credit-uplift-1",training_sha256=_hash(path),plan_sha256=_hash(plan),
                       train_ids=[r["id"] for r in rows],retained_ids=[r["id"] for r in fitrows],
                       context=mdl.context.tolist(),uplift=mdl.uplift.tolist(),strata=by,
                       oracle_any_teacher=int(np.max(variant,axis=1).sum()),offers=len(rows),
                       teacher_legal=dict(zip(TEACHERS,map(int,variant.sum(0)))),
                       gains=dict(zip(TEACHERS[1:],map(int,(variant[:,1:]>variant[:,0:1]).sum(0)))),
                       losses=dict(zip(TEACHERS[1:],map(int,(variant[:,1:]<variant[:,0:1]).sum(0)))),
                       leave_one_release_out=diagnostic,
                       movements=[dict(id=r["id"],by_teacher={k:movement(r,k) for k in TEACHERS[1:]}) for r in rows],
                       rule=dict(min_gain=MIN_GAIN,min_help=MIN_HELP,prior=PRIOR,scale=SCALE.tolist())))


class Deployment(Teacher):
    def step(self,tick,eye,x,q,reference,base):
        early=self.get_base(tick,x,q,reference,self.priors.home)
        return super().step(tick,eye,x,q,reference,early)


def play(path,plan,feeds,output,limit,cap):
    planv=_read(plan); learner=_read(path)
    test=planv["held"][:limit]
    if (learner["plan_sha256"]!=_hash(plan) or
            {c["id"] for c in test}&set(learner["train_ids"]) or not test):
        raise ValueError("Probe releases overlap the frozen training")
    selector=Selector(learner["context"],learner["uplift"])
    baseline=_read(STOCK)
    if baseline["carry_sha256"] is not None or baseline["mode"]!="frozen":
        raise ValueError("Baseline must have the same original causal memory")
    controls={k["key"]:(k,r) for k,r in zip(baseline["cards"],baseline["rows"],strict=True)}
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    ready=np.array(_read(mpc.BANK.with_suffix(".json"))["ready"])
    motor=servo.reload(servo.ROOT/"prior.npz")
    shots=mpc.measured_launches(feeds)
    outputv=dict(schema="tt44-paired-credit-game-transfer-1",plan_sha256=_hash(plan),
                 learner_sha256=_hash(path),archived_stock_sha256=_hash(STOCK),cap=cap,
                 cards=test,ready=ready.tolist(),rows=[])
    with mpc.centered(ready):
        a=mpc.arena(); camera,rig,stride,lo,hi=mpc.resources(a)
        with patch.object(mpc,"Fragments",lambda file,e:Carried(file,e,practice=[course.PRACTICE],held={c["id"] for c in test})):
            actor=Deployment(mpc.BANK,raw,variant="elastic",budget=28,policy="stock",motor=motor,selector=selector)
        try:
            for c in test:
                original,b=controls[c["key"]]
                if original!=c or shots[c["id"]]!=b["shot"]:
                    raise RuntimeError("Mismatched evaluator launch")
                r=mpc.play(a,camera,rig,stride,lo,hi,ready,c,shots[c["id"]],raw,actor,
                           course.OPTIC,"frozen",cap)
                for leg,item in enumerate(r["placements"]["requested"]):
                    if item["target"]!=list(draw(c,leg,"random",None)[0]):
                        raise RuntimeError("The deployed actor changed the opponent's random draw")
                r["choices"]=actor.approaches
                ds=list(zip(r["actions"],b["actions"]))
                r["first_changed_tick"]=next((i for i,(x,y) in enumerate(ds) if x!=y),None)
                r["baseline_own"]=b["score"]["learner"]
                r["original_baseline_actions_complete"]=bool(len(b["actions"])<=len(r["actions"]) and
                                                              r["first_changed_tick"] is None)
                a0,b0=r["placements"]["delivered"][:1],b["placements"]["delivered"][:1]
                r["shared_first_delivery"]=len(a0)==len(b0) and all(
                    all(same_trace(x[k],y[k]) for k in ("far_contact","near_bounce")) for x,y in zip(a0,b0))
                outputv["rows"].append(r)
                _write(output,outputv)
                print(c["id"],r["baseline_own"],r["score"]["learner"],
                      [z["mode"] for z in actor.approaches],r["first_changed_tick"],flush=True)
        finally:
            actor.close();camera.close();a.close()
    outputv["complete"]=True
    _write(output,outputv)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("fit","play"))
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--training",type=Path,default=ROOT/"training.json.gz")
    p.add_argument("--policy",type=Path,default=ROOT/"uplift.json")
    p.add_argument("--output",type=Path,default=ROOT/"held.json.gz")
    p.add_argument("--feeds",type=Path,default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--limit",type=int,default=4)
    p.add_argument("--cap",type=int,default=240)
    a=p.parse_args()
    fit(a.training,a.plan,a.policy) if a.phase=="fit" else play(a.policy,a.plan,a.feeds,a.output,a.limit,a.cap)
