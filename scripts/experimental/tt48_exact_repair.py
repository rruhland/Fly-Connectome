"""Transport physically scored early joint corrections to exact new season offers."""

import argparse
from collections import deque
from pathlib import Path
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_carry_joints import Carried
from .tt39_opportunity_map import projected_height
from .tt40_long_low import same_trace
from .tt44_continuous_credit import ARM_CUE, ENVELOPE, ROOT as PRACTICE, sensed_look
from .tt44_continuous_credit_fit import Local, input_state
from .tt44_paired_credit import State, Teacher, prefix, suffix, verify_restore

ROOT=Path("artifacts/tt48_exact_repair")
SOURCE=season.ROOT/"before.json.gz"
MODEL=PRACTICE/"policy.json"
# Short held ordinary glances, a moderate-low glance, and a normal scored
# control. Only the previous archived outcomes stratify the probes.
POOLS=((4,"ordinary miss"),(23,"ordinary miss"),(7,"moderate miss"),(5,"easy control"))


class Repair(season.Player):
    def __init__(self,*a,readout,examples,**kw):
        super().__init__(*a,**kw)
        self.readout=readout
        self.examples=examples
        self.mode="stock"
        self.policy="stock"
        self.selector=None
        self.forced=None
        self.keep_until=-.53

    def begin(self):
        super().begin()
        self.sensed=False
        self.inbound=deque(maxlen=7)
        self.context=None
        self.choice=None
        self.approaches=[]
        self.correct=None
        self.picked=[]

    def get_base(self,tick,x,q,ref,home):
        base=Teacher.get_base(self,tick,x,q,ref,home)
        if x is None:
            return base
        if x[4]>.9:
            self.correct=None
        if x[4]>=-.9 or not self.keep_until<x[1]<ARM_CUE[1]:
            return base
        if self.correct is None:
            z=input_state(x,q,ref)
            if self.mode=="learned":
                self.correct=self.readout.get(z)
                borrowed=None
            elif self.mode=="nearest-demo":
                x0,u,borrowed=min(self.examples,key=lambda k:np.mean((z-k[0])**2))
                self.correct=u.copy()
            elif self.mode=="forced":
                self.correct=np.asarray(self.forced).copy()
                borrowed=None
            else:
                self.correct=np.zeros(6)
                borrowed=None
            self.picked.append(dict(tick=tick,source_case=borrowed,parameters=self.correct.tolist(),
                                    context=z.tolist()))
        k=self.correct
        if not np.any(k):
            return base
        look=sensed_look(x,k[0]*ENVELOPE[0])
        future,w=self.raw.get(look,q)
        lead=base if future is None else home+w*(future-home)
        frame=np.clip((x[1]-self.keep_until)/.31,0.,1.)*np.clip((1.32-x[1])/.47,0.,1.)
        offset=k[1:]*ENVELOPE[1:]
        lag=ref[:5]-q[:5]
        attenuate=np.where(offset*lag>0.,1.-.38*np.clip(abs(lag)/.28,0.,1.),1.)
        delta=np.clip(lead[:5]-base[:5]+offset*attenuate,-.68,.68)
        z=base.copy()
        z[:5]+=frame*delta
        self.diag["borrowed_preparation_ticks"]+=int(np.max(abs(frame*delta))>1e-6)
        return z


def docket(d):
    plan=[]
    for index,label in POOLS:
        c,r=d["cards"][index],d["rows"][index]
        offers=r["placements"]["delivered"]
        if label=="easy control":
            options=[(i,o) for i,o in enumerate(offers[:3]) if o.get("receive_line") and o["near_landed"]]
            leg,offer=max(options,key=lambda p:projected_height(p[1]["receive_line"]))
        else:
            leg,offer=len(offers)-1,offers[-1]
            if not offer["near_bounce"] or offer["near_landed"] or not offer["near_contact"]:
                raise ValueError("Changed original challenge")
        z=projected_height(offer["receive_line"])
        t=next(e["time"] for e in r["events"] if e.get("collider")=="rival_racket"
               and offer["far_contact"]["time"]<=e["time"]<offer["near_bounce"]["time"])
        plan.append(dict(id=c["id"],leg=leg,index=index,card=c,role=label,
                         tick=int(np.ceil(t/mpc.DT-1e-7)),actual_projected_m=z,
                         archive_legal=offer["near_landed"]))
    return plan


def run(out,limit,full_observer=False):
    d=_read(SOURCE)
    model=_read(MODEL)
    roster=docket(d)[:limit]
    ids={r["id"] for r in roster}
    if d["mode"]!="frozen" or d["carry_sha256"]!=_hash(season.CARRY) or ids&{r["id"] for r in model["rows"]}:
        raise ValueError("Need new physical trajectories on the original fixed season checkpoint")
    readout=Local(model["rows"],[z["context"] for z in model["easy"]])
    examples=[(np.asarray(r["context"]),np.asarray(t["u"]),r["id"]) for r in model["rows"]
      for t in r["trials"] if t["reward"]>r["stock"] and r["id"] in (9519,4283)]
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    start=_read(season.CARRY)
    boot=np.asarray(d["ready"])
    result=dict(schema="tt48-exact-strong-correction-transfer-1",source_sha256=_hash(SOURCE),
                learner_sha256=_hash(MODEL),carry_sha256=_hash(season.CARRY),roster=roster,rows=[],
                full_observer_prefix=full_observer,
                scope="Assessment only. Physical archive provides the common reset and labels, not arm observations.")
    with mpc.centered(boot):
        a=mpc.arena();camera,rig,stride,lo,hi=mpc.resources(a)
        with patch.object(mpc,"Fragments",lambda f,e:Carried(f,e,practice=[course.PRACTICE],held=ids)):
            actor=Repair(mpc.BANK,raw,readout=readout,examples=examples,coverage=fresh.shot.Coverage(_read(fresh.RISK)),
                         budget=d["budget"],variant=d["variant"])
        course.resume(actor,season.CARRY.with_suffix("").with_suffix(".npz"),start)
        try:
            for c in roster:
                before=process_time()
                old=d["rows"][c["index"]]
                if actor.priors.digest()!=old["memory"]["before"]:
                    raise ValueError("Changed physical practice memory")
                memory=prefix(a,c["card"],old,c["tick"],camera,a._arm_ids[1][2],window=None if full_observer else 22)
                saved=State(a)
                verify_restore(a,saved,old,min(c["tick"]+95,len(old["actions"])),c)
                v={}
                for mode in ("stock","learned","nearest-demo"):
                    saved.restore(a)
                    actor.mode=mode
                    r=suffix(a,camera,rig,stride,lo,hi,actor,memory,c,old,boot)
                    r["choice"]=actor.picked
                    v[mode]=r
                    if mode=="learned" and all(not any(p["parameters"]) for p in r["choice"]):
                        if r["controls"]!=v["stock"]["controls"] or r["events_sha256"]!=v["stock"]["events_sha256"]:
                            raise ValueError("The zero-action treatment behaved differently from stock")
                result["rows"].append(dict(c,variants=v,native_restore_verified=True,cpu_s=process_time()-before,
                    baseline_matches_archived_commands=old["actions"][c["tick"]:c["tick"]+len(v["stock"]["controls"])]
                                                       ==v["stock"]["controls"]))
                _write(out,result)
                print(c["id"],c["leg"],c["role"],[(k,r["landed"],r["touched"],
                       r["choice"][0]["source_case"] if r["choice"] else None) for k,r in v.items()],flush=True)
        finally:
            actor.close();camera.close();a.close()
    result["complete"]=True
    _write(out,result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"transfer.json.gz")
    p.add_argument("--limit",type=int,default=len(POOLS))
    p.add_argument("--full-observer",action="store_true")
    a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    run(a.output,a.limit,a.full_observer)
