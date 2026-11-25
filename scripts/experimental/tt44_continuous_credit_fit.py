"""Amortize achieved early continuous movements using physical legal-return contrasts."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt43_temporal_servo as servo
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_carry_joints import Carried
from .tt40_long_low import same_trace
from .tt44_continuous_credit import ARM_CUE, PAIRED, PLAN, ROOT, Continuous
from .tt44_paired_credit import sources, suffix
from .tt44_paired_credit_fit import STOCK
from .tt44_continuous_credit import reset_setup

ZERO=np.zeros(6)


def input_state(x,q,ref):
    return np.r_[x/[.18,.8,.22,.7,1.45,1.03],q[:5]/[.45,.36,.36,.53,.56],
                 (ref[:5]-q[:5])/.18]


def first_public(sensory):
    for row in sensory:
        if ARM_CUE[0]<row[2]<ARM_CUE[1] and row[5]<-.9:
            t=int(row[0])
            return t,input_state(np.asarray(row[1:7]),np.asarray(row[7:13]),np.asarray(row[13:19]))
    return None,None


class Local:
    """One vote per physically distinct offer, not one independent vote per perturbation."""
    def __init__(self,rows,easy):
        self.rows=rows
        self.easy=np.asarray(easy).reshape(-1,16)
        self.winners=[np.asarray(t["u"]) for r in rows for t in r["trials"] if t["reward"]>r["stock"]]

    def get(self,z):
        if not self.winners:
            return ZERO.copy()
        distance=np.array([np.mean((z-np.asarray(r["context"]))**2) for r in self.rows])
        near=np.exp(-3.*distance)
        easy=np.exp(-3.*np.mean((self.easy-z)**2,axis=1))
        best_score,best=.07,ZERO
        for k in self.winners:
            pos=[];neg=[]
            for r in self.rows:
                sim=[np.exp(-1.2*np.mean(((k-np.asarray(t["u"]))/.45)**2)) for t in r["trials"]]
                effect=[t["reward"]-r["stock"] for t in r["trials"]]
                pos.append(max([s for s,e in zip(sim,effect) if e>0],default=0.))
                # A failed moved attempt on a ball already missed by stock
                # does not regress score, but does dilute nearby rescue support.
                neg.append(max([s for s,e in zip(sim,effect) if e<=0],default=0.))
            p=np.asarray(pos)*near
            n=np.asarray(neg)*near
            w=float(np.max(p))
            if w<.63 or (len(easy) and np.max(easy)>.91 and np.max(easy)>w+.19):
                continue
            value=float((p.sum()-.070*n.sum())/(1.25+p.sum()+n.sum()))
            if value>best_score:
                best_score,best=value,k
        return np.array(best)


def fit(new,paired,out):
    data=_read(new); base=_read(paired)
    if (not data.get("complete") or data["outbound"] or data.get("keep_to",-.53)!=-.53
            or data["pairs_sha256"]!=_hash(paired)):
        raise ValueError("Fit only complete early-after-far practice")
    packets=[]
    guard=[]
    for row in base["rows"]:
        if row["stratum"]=="easy":
            _,z=first_public(row["variants"]["stock"]["sensory"])
            if z is not None:
                guard.append(dict(id=row["id"],context=z.tolist()))
    for row in data["rows"]:
        origin=next(b["variants"]["stock"] for b in base["rows"] if b["id"]==row["id"])
        tick,z=first_public(origin["sensory"])
        if z is None:
            continue
        trials=[]
        for run in row["trials"]:
            k,z1=first_public(run["sensory"])
            if k!=tick or z1 is None or not np.array_equal(z,z1):
                raise ValueError(f"Action changed between paired public choice states {row['id']}")
            trials.append(dict(u=run["parameters"],reward=int(run["landed"]),touched=run["touched"]))
        packets.append(dict(id=row["id"],context=z.tolist(),stock=int(origin["landed"]),trials=trials))
    local=Local(packets,[r["context"] for r in guard])
    _write(out,dict(schema="tt44-continuous-local-effect-1",paired_sha256=_hash(paired),
                    practice_sha256=_hash(new),rows=packets,easy=guard,policy_examples=len(local.winners),
                    outcomes=dict(offers=len(data["rows"]),trials=sum(len(c["trials"]) for c in data["rows"]),
                                  stock_legal=sum(r["stock_landed"] for r in data["rows"]),
                                  any_new_legal=sum(bool(r["modified_landed"]) for r in data["rows"]))))


class Amortized(Continuous):
    def begin(self):
        super().begin()
        self.selection=None
        self.receipt=[]

    def get_base(self,tick,x,q,reference,home):
        if x is not None and x[4]>.90:
            self.selection=None
            self.parameter=np.zeros(6)
        if x is not None and x[4]<-.90 and self.selection is None and ARM_CUE[0]<x[1]<ARM_CUE[1]:
            z=input_state(x,q,reference)
            self.selection=self.local.get(z)
            self.parameter=self.selection.copy()
            self.receipt.append(dict(tick=tick,context=z.tolist(),parameter=self.selection.tolist()))
        return super().get_base(tick,x,q,reference,home)


class Deployment(Amortized):
    def step(self,tick,eye,x,q,reference,base):
        return super().step(tick,eye,x,q,reference,self.get_base(tick,x,q,reference,self.priors.home))


def trial(mode,path,plan,feeds,out,limit,paired):
    learned=_read(path); spec=_read(plan); base=_read(paired)
    if learned["paired_sha256"]!=_hash(paired):
        raise ValueError("The chosen practice corpus changed")
    local=Local(learned["rows"],[r["context"] for r in learned["easy"]])
    if not local.winners:
        raise ValueError("No physically rewarded new movement to transfer")
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    home=np.asarray(_read(mpc.BANK.with_suffix(".json"))["ready"])
    model=servo.reload(servo.ROOT/"prior.npz")
    score_archive=_read(STOCK)
    delivery_map={c["key"]:(c,r) for c,r in zip(score_archive["cards"],score_archive["rows"],strict=True)}
    excluded={r["id"] for r in spec["train"]}
    reserved={c["id"] for k in ("practice","assessment") for c in _read(Path("artifacts/tt44_combined/fresh.json"))[k]}
    cards=[c for c in _read(course.PLAN)["assessment"][6:10] if c["id"] not in excluded|reserved][:limit]
    receipt=dict(schema="tt44-continuous-transfer-1",mode=mode,model_sha256=_hash(path),
                 plan_sha256=_hash(plan),rows=[],ready=home.tolist())
    with mpc.centered(home):
        a=mpc.arena();camera,rig,stride,lo,hi=mpc.resources(a)
        clazz=Deployment if mode=="held" else Amortized
        with patch.object(mpc,"Fragments",lambda f,e:Carried(f,e,practice=[course.PRACTICE],held={c["id"] for c in cards})):
            actor=clazz(mpc.BANK,raw,variant="elastic",budget=28,policy="stock",motor=model)
        actor.local=local
        try:
            if mode=="easy":
                previous=sources()
                for r in base["rows"]:
                    if r["stratum"]!="easy":
                        continue
                    _,z=first_public(r["variants"]["stock"]["sensory"])
                    if z is None or not np.any(local.get(z)):
                        receipt["rows"].append(dict(id=r["id"],selected_continuous=False,
                                                    baseline_legal=True,legal=True,
                                                    method="identity decision at a replay-verified fork"))
                        continue
                    _,d=previous[r["source"]]
                    old=d["rows"][r["index"]]
                    current,readings,saved=reset_setup(a,r,old,camera,home,False)
                    saved.restore(a)
                    v=suffix(a,camera,rig,stride,lo,hi,actor,readings,current,old,home)
                    receipt["rows"].append(dict(id=r["id"],selected_continuous=True,baseline_legal=True,
                                                legal=v["landed"],run=v,choices=actor.receipt))
                    _write(out,receipt)
            elif mode=="held":
                shots=mpc.measured_launches(feeds)
                receipt["cards"]=cards
                for c in cards:
                    card,ref=delivery_map[c["key"]]
                    if card!=c or shots[c["id"]]!=ref["shot"]:
                        raise ValueError("Different official evaluation tape")
                    r=mpc.play(a,camera,rig,stride,lo,hi,home,c,shots[c["id"]],raw,actor,
                               course.OPTIC,"frozen",240)
                    r["id"]=c["id"]
                    r["choices"]=actor.receipt
                    r["fixed_stock"]=ref["score"]["learner"]
                    r["first_altered_control"]=next((i for i,(u,v) in enumerate(zip(r["actions"],ref["actions"]))
                                                       if u!=v),None)
                    aa,bb=(z["placements"]["delivered"][:1] for z in (r,ref))
                    r["shared_first_delivery"]=len(aa)==len(bb) and all(
                          same_trace(p[k],q[k]) for p,q in zip(aa,bb) for k in ("far_contact","near_bounce"))
                    receipt["rows"].append(r)
                    _write(out,receipt)
                    print(c["id"],r["fixed_stock"],r["score"]["learner"],
                          sum(bool(np.any(z["parameter"])) for z in r["choices"]),flush=True)
        finally:
            actor.close();camera.close();a.close()
    receipt["complete"]=True
    _write(out,receipt)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("fit","easy","held"))
    p.add_argument("--training",type=Path,default=ROOT/"training.json.gz")
    p.add_argument("--paired",type=Path,default=PAIRED)
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--policy",type=Path,default=ROOT/"policy.json")
    p.add_argument("--output",type=Path)
    p.add_argument("--feeds",type=Path,default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--limit",type=int,default=3)
    a=p.parse_args()
    if a.phase=="fit":
        fit(a.training,a.paired,a.policy)
    else:
        out=a.output or ROOT/f"{a.phase}.json.gz"
        trial(a.phase,a.policy,a.plan,a.feeds,out,a.limit,a.paired)
