"""Attempt release-disjoint RGB discrepancy overrides on matched native openings."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_safe_drive import PoseClearance
from .tt37_opponent_audit import CourtClearance, guarded
from .tt39_carry_joints import Carried
from .tt39_opportunity_map import projected_height
from .tt44_paired_credit import State, prefix, suffix, verify_restore
from .tt48_contact_patch import ROOT, ease
from .tt48_edge_evidence_fit import COLLECTED, Evidence, pair_features
from .tt48_rim_recovery import DIRECT, PLAN as RIM_PLAN, STOCK, Rim, alignment, remap

PLAN=ROOT/"discrepancy-pilot.protocol.json"
CASES=((4,"miss"),(23,"miss"),(12,"miss"),(5,"retain"))


class Clause(Rim):
    def __init__(self,*args,readout,**kw):
        super().__init__(*args,**kw)
        self.readout=readout
        self.mode="stock"
        self.guard=PoseClearance(self.engine.twin)

    def begin(self):
        super().begin()
        self.approaches=[]
        self.override=[]
        self.sensed=False

    def get_base(self,tick,x,q,ref,home):
        if x is not None and x[4]<-.9 and x[1]<-.90:
            self.sensed=True
        if x is not None and (x[4]<-.9 or (self.sensed and x[1]<.50)):
            goal,w=self.raw.get(x,q)
            if goal is not None:
                return home+w*(goal-home)
        return home

    def search(self,tick,x,q,ref,base):
        now=len(self.notes)
        watch=(-1.13<x[1]<-.49 and x[4]<-.9 and tick-self.last_edge>7
               and self.mode!="stock")
        self.capture=[] if watch else None
        try:
            # Keep the already established temporal + outgoing ranker. Do
            # not enable anything from the separate sealed rim policy.
            season.Player.search(self,tick,x,q,ref,base)
            pool=self.capture
        finally:
            self.capture=None
        if not pool or len(self.notes)==now or self.notes[-1].get("legal_trials",0):
            return
        self.last_edge=tick
        seed=[]
        for g,f in sorted(pool,key=lambda pair: abs(pair[1][7]-1.0)*8.+abs(pair[1][6]+1.56)):
            if any(np.max(abs(g-previous))<.05 for previous,_ in seed):
                continue
            seed.append((g,f))
            if len(seed)==3:
                break
        angle=alignment(self.engine,[g for g,_ in seed])
        opt=np.r_[x,q[:5],ref[:5]-q[:5]]
        frames=(1+np.arange(mpc.HORIZON))*mpc.DT
        proposals=[]
        roster=[]
        for (g,f),old_align in zip(seed,angle,strict=True):
            if not np.isfinite(old_align) or (old_align>.88 and f[3]>.9 and f[4]>.8):
                continue
            eta=float(np.clip((x[1]-f[6])/-x[4],.065,.29))
            for donor in self.corrections:
                delta=remap(donor["delta"],donor["flank"],f[5])
                for rate in (.68,1.03):
                    move=g+ease(frames/max(.04,.84*eta))[:,None]*rate*delta
                    proposals.append(move)
                    roster.append((f,eta,rate,donor["release"],rate*delta[3]))
        if not proposals:
            return
        _,legal,trace,_,effect=DIRECT.trial(self,proposals,traces=True)
        # Inspect only likely contacts, and never treat an unobserved
        # collision as direct evidence for a landing.
        indices=np.flatnonzero((effect[:,1]>.5)&(effect[:,8]>.060)&(effect[:,8]<.081)&~legal)
        indices=sorted(indices,key=lambda i:-(effect[i,3]+.65*effect[i,4]))[:8]
        if not indices:
            return
        angles=alignment(self.engine,[proposals[i] for i in indices])
        votes=[]
        for j,normal in zip(indices,angles,strict=True):
            if not np.isfinite(normal):
                continue
            f,eta,rate,donor,wrist=roster[j]
            features=pair_features(opt,dict(virtual_feedback=f),
                                   dict(virtual_feedback=effect[j],virtual_alignment=normal),wrist)
            item=dict(id=-1,x=features.tolist(),nominal=effect[j,2:5].tolist(),
                      nominal_p=effect[j,5:8].tolist(),weak=True,virtual_contact=True,
                      native_legal=bool(legal[j]),face=float(normal))
            result=self.readout.assess(item,self.mode)
            if not result["accepted"]:
                continue
            reached=trace[j,:,6:11]
            lease=min(mpc.HORIZON,max(mpc.LEASE,int(np.ceil(eta/mpc.DT))+3))
            path=reached[:lease]
            if np.isfinite(path).all(axis=1).sum()<max(2,lease-4):
                continue
            tested=path[np.isfinite(path).all(axis=1)]
            if any(self.guard.violations(point) for point in tested):
                continue
            votes.append((result["p_physically_earned"],int(j),lease,donor,rate,result))
        self.override.append(dict(tick=tick,candidates=len(indices),qualified=len(votes),issued=False))
        if votes:
            _,i,lease,release,rate,detail=max(votes,key=lambda z:z[0])
            self.best=proposals[i]
            self.tick=tick
            self.lease=lease
            self.next=tick+lease
            self.override[-1].update(issued=True,training_example=release,gain=rate,
                                     lease=lease,direction=self.mode,evidence=detail)
            self.diag["edge_override_issued"]+=1


def schedule(out):
    fixed, bank=_read(STOCK),_read(COLLECTED)
    if not bank["qualifies"]:
        raise ValueError("Do not test an unqualified discrepancy law")
    excluded={r["id"] for r in bank["rows"]}
    roster=[]
    for i,name in CASES:
        c,r=fixed["cards"][i],fixed["rows"][i]
        f=r["placements"]["delivered"][0]
        if c["id"] in excluded or f["near_landed"]!=(name=="retain"):
            raise ValueError("The matched opening docket changed")
        t=next(e["time"] for e in r["events"] if e.get("collider")=="rival_racket"
               and f["far_contact"]["time"]<=e["time"]<f["near_bounce"]["time"])
        roster.append(dict(id=c["id"],card=c,index=i,leg=0,role=name,
                           tick=int(np.ceil(t/mpc.DT-1e-7)),
                           height=projected_height(f["receive_line"])))
    _write(out,dict(roster=roster,baseline_sha256=_hash(STOCK),model_sha256=_hash(COLLECTED),
                    carry_sha256=_hash(season.CARRY),controller_sha256=_hash(Path(__file__)),
                    earlier_protocol_sha256=_hash(RIM_PLAN),cohorts=["stock","learned","reverse"],
                    scope="Targeted, actual same-opening perturbation. Old outcome used only to stratify openings in this diagnostic, not used by the actor. Retain normal capped/uncapped whole-game evaluation separately."))


def run(protocol,out):
    plan,stock,model,rim=_read(protocol),_read(STOCK),_read(COLLECTED),_read(RIM_PLAN)
    if (plan["baseline_sha256"]!=_hash(STOCK) or plan["model_sha256"]!=_hash(COLLECTED)
            or plan["controller_sha256"]!=_hash(Path(__file__))
            or plan["carry_sha256"]!=_hash(season.CARRY) or not model["qualifies"]):
        raise ValueError("Changed matched-pilot conditions")
    receipt=dict(protocol_sha256=_hash(protocol),roster=plan["roster"],rows=[],complete=False)
    if out.exists():
        old=_read(out)
        if any(old[k]!=v for k,v in receipt.items() if k not in ("rows","complete")):
            raise ValueError("Different experimental docket")
        receipt=old
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    ready=np.asarray(stock["ready"])
    fit=Evidence(model["rows"])
    with mpc.centered(ready):
        a=mpc.arena(); camera,rig,stride,lo,hi=mpc.resources(a)
        shadow=mpc.arena(detail="cad")
        guard=CourtClearance(shadow.model,articulated=True)
        with patch.object(mpc,"Fragments",lambda path,engine: Carried(
                path,engine,practice=[course.PRACTICE],held={r["id"] for r in plan["roster"]})):
            actor=Clause(mpc.BANK,raw,variant="elastic",budget=stock["budget"],readout=fit,
                         coverage=fresh.shot.Coverage(_read(fresh.RISK)),corrections=rim["corrections"])
        try:
            for group in plan["roster"][len(receipt["rows"]):]:
                old=stock["rows"][group["index"]]
                course.resume(actor,season.CARRY.with_suffix("").with_suffix(".npz"),_read(season.CARRY))
                readings=prefix(a,group["card"],old,group["tick"],camera,a._arm_ids[1][2],window=None)
                saved=State(a)
                verify_restore(a,saved,old,min(group["tick"]+95,len(old["actions"])),group)
                arms={}
                for mode in plan["cohorts"]:
                    saved.restore(a)
                    actor.mode=mode
                    # CAD is a separate observer on actual trials, never a
                    # sensor or an edited physical environment.
                    guarded(a,guard)
                    mark=float(a.data.time)
                    v=suffix(a,camera,rig,stride,lo,hi,actor,readings,group,old,ready)
                    v.update(overrides=actor.override,clearance=a.clearance_result(),
                             referee=a.events[len(saved.dynamic["events"]):])
                    if v["clearance"]["samples"]!=round((a.data.time-mark)/a.model.opt.timestep):
                        raise ValueError("Independent clearance sampling missed a native step")
                    arms[mode]=v
                if arms["stock"]["landed"]!=(group["role"]=="retain"):
                    raise ValueError("Short baseline no longer matches the archived opening outcome")
                for mode in ("learned","reverse"):
                    if not arms[mode]["diagnostics"].get("edge_override_issued"):
                        if arms[mode]["controls"]!=arms["stock"]["controls"]:
                            raise ValueError("Unissued evidence perturbed stock action")
                a.sample_clearance(False)
                receipt["rows"].append(dict(group,arms=arms,restored=True))
                receipt["complete"]=len(receipt["rows"])==len(plan["roster"])
                _write(out,receipt)
                print(group["id"],[(m,v["landed"],sum(e["issued"] for e in v["overrides"]),
                   v["clearance"]["penetration_samples"]) for m,v in arms.items()],flush=True)
        finally:
            actor.close(); shadow.close(); camera.close(); a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("task",choices=("plan","run"))
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--out",type=Path,default=ROOT/"discrepancy-pilot.json.gz")
    a=p.parse_args()
    schedule(a.plan) if a.task=="plan" else run(a.plan,a.out)
