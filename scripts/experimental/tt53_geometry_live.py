"""Replace four existing fallback samples after the stock native screen finds no legal shot."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from . import tt53_signed_practice as geometry
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent import draw
from .tt39_carry_joints import Carried
from .tt48_contact_patch_report import replay
from .tt48_rim_recovery import STOCK

PLAN=geometry.ROOT/"geometry-live.protocol.json"
OUT=geometry.ROOT/"geometry-live.json.gz"
PROOF=geometry.ROOT/"geometry-report.json"
N=4


class Basis(season.Player):
    def __init__(self,*args,library,stride,lo,hi,**kwargs):
        super().__init__(*args,**kwargs)
        self.library=library
        self.stride,self.lo,self.hi=stride,lo,hi
        self.decision=None

    def begin(self):
        super().begin()
        self.additions=[]

    def search(self,tick,x,q,ref,base):
        self.decision=dict(tick=tick,x=x,q=q,ref=ref,anchor=None,first_long=False,empty=False,inserted=None)
        try:
            super().search(tick,x,q,ref,base)
            d=self.decision
            if d["inserted"] is not None:
                note=d["note"]
                if self.tick==tick and self.best is not None:
                    got=[key for key,g in d["inserted"] if np.array_equal(self.best,g)]
                    if got:
                        note["issued"]=got[0]
                        self.diag["learned_basis_issued"]+=1
                self.additions.append(note)
        finally:
            self.decision=None

    def trial(self,goals,*,long=False,traces=False):
        d=self.decision
        if d is not None and not long and not traces and len(goals)==12 and d["empty"] and d["anchor"] is not None:
            if d["inserted"] is not None or not isinstance(goals,list):
                raise ValueError("Ordinary fallback candidate contract changed")
            raw=d["anchor"]
            ref=d["ref"]
            r=ref.copy()
            normalized=[]
            for g in raw:
                u=np.clip((np.clip(g,self.lo,self.hi)-r)/self.stride,-.8,.8)
                r=np.clip(r+u*self.stride,self.lo,self.hi)
                normalized.append(u)
            normalized=np.asarray(normalized)
            basis,detail=geometry.local_basis(self.library,d["x"],d["q"],ref,normalized)
            if detail["distances"][0]<8.:
                proposed=[]
                for k,coeff in geometry.REPEATS:
                    u=geometry.issue(normalized,basis,geometry.ALPHA*np.asarray(coeff),self.stride,self.lo,self.hi,ref)
                    r=ref.copy()
                    q=[]
                    for motor in u:
                        r=np.clip(r+motor*self.stride,self.lo,self.hi)
                        q.append(r.copy())
                    proposed.append((k,np.asarray(q)))
                # Replace only the last four already-budgeted generic
                # fallback samples. Their RNG already ran as usual.
                goals[-4:]=[g for _,g in proposed]
                d["inserted"]=proposed
                d["note"]=dict(tick=d["tick"],issued=None,candidates=4,stock_first_screen_legal=0,
                               donor_ids=detail["donor_ids"],nearest_distance=detail["distances"][0])
                self.diag["learned_basis_searches"]+=1
                self.diag["learned_basis_slots"]+=4
        result=super().trial(goals,long=long,traces=traces)
        if d is not None:
            if not long and not traces and len(goals)==self.budget and d["anchor"] is None:
                impact=np.flatnonzero(result[4][:,1]>.5)
                if len(impact):
                    ii=int(impact[np.argmin(result[0][impact])])
                    d["anchor"]=np.asarray(goals[ii]).copy()
                else:
                    d["anchor"]=None
            if long and not d["first_long"]:
                d["first_long"]=True
                d["empty"]=not np.any(result[1])
        return result


def schedule(out):
    reference,training,evidence=_read(STOCK),_read(geometry.LINES),_read(PROOF)
    learned=evidence["unique_releases_any_fork"]["learned"]
    sham=evidence["unique_releases_any_fork"]["sham"]
    cards=reference["cards"][:N]
    if (not evidence["complete"] or len(set(learned)-set(sham))<2 or set(sham)-set(learned)
            or any(c["id"] in {r["id"] for r in training["cases"]} for c in cards)
            or len({c["id"] for c in cards})!=N):
        raise ValueError("No independent candidate geometry proof")
    _write(out,dict(cards=cards,cap=reference["cap"],stock_sha256=_hash(STOCK),
                    training_sha256=_hash(geometry.LINES),proof_sha256=_hash(PROOF),
                    code_sha256=_hash(Path(__file__)),carry_sha256=_hash(season.CARRY),
                    risk_sha256=_hash(fresh.RISK),rule="First four preexisting independent frozen official assessment streams in order. Original native short and long screens must find zero legal candidate before replacing the last four of its twelve already-budgeted failure-refinement samples. No extra trial rows, different opponent, known upcoming successful motion, changed control cap or updated mechanics."))


def run(protocol,out):
    plan,reference,teaching=map(_read,(protocol,STOCK,geometry.LINES))
    for k,p in {"stock_sha256":STOCK,"training_sha256":geometry.LINES,"proof_sha256":PROOF,
                "code_sha256":Path(__file__),"carry_sha256":season.CARRY,"risk_sha256":fresh.RISK}.items():
        if plan[k]!=_hash(p):
            raise ValueError("Changed frozen live candidate restriction")
    result=dict(complete=False,protocol_sha256=_hash(protocol),rows=[])
    if out.exists():
        result=_read(out)
        if result["protocol_sha256"]!=_hash(protocol):
            raise ValueError("Different live schedule")
    cards=plan["cards"]
    with np.load(mpc.BANK) as raw_data:
        raw=mpc.DeepField(raw_data,"local")
    ready=reference["ready"]
    train_docs=_read(season.CARRY)
    with mpc.centered(ready):
        a=mpc.arena();lens,rig,stride,lo,hi=mpc.resources(a)
        with patch.object(mpc,"Fragments",lambda b,e: Carried(b,e,practice=[course.PRACTICE],
                         held={c["id"] for c in cards})):
            actor=Basis(mpc.BANK,raw,variant=reference["variant"],budget=reference["budget"],
                        coverage=fresh.shot.Coverage(_read(fresh.RISK)),
                        library=geometry.source_bank(teaching["cases"],-1),stride=stride,lo=lo,hi=hi)
        try:
            course.resume(actor,season.CARRY.with_suffix("").with_suffix(".npz"),train_docs)
            for i in range(len(result["rows"]),len(cards)):
                c=cards[i]; old=reference["rows"][i]
                if actor.priors.digest()!=old["memory"]["before"]:
                    raise ValueError("Different frozen motor memory")
                r=mpc.play(a,lens,rig,stride,lo,hi,ready,c,old["shot"],raw,actor,course.OPTIC,
                           "frozen",plan["cap"])
                for leg,record in enumerate(r["placements"]["requested"]):
                    if record["target"]!=list(draw(c,leg,"random",None)[0]):
                        raise ValueError("Changed physically reacted target stream")
                d=next((j for j,(u,v) in enumerate(zip(old["actions"],r["actions"])) if u!=v),
                       None if len(old["actions"])==len(r["actions"]) else min(len(old["actions"]),len(r["actions"])))
                found=[]
                for u,v in zip(old["placements"]["delivered"],r["placements"]["delivered"]):
                    if u["near_landed"]==v["near_landed"]:
                        continue
                    if all(digest_json(u[k])==digest_json(v[k]) for k in ("far_contact","near_bounce")):
                        found.append(dict(leg=u["leg"],stock_landed=u["near_landed"],learned_landed=v["near_landed"]))
                if d is None and digest_json(old["events"])!=digest_json(r["events"]):
                    raise ValueError("Different native events despite identical controls")
                if d is not None and digest_json([e for e in old["events"] if e["step"]<=d])!=digest_json(
                        [e for e in r["events"] if e["step"]<=d]):
                    raise ValueError("Same-ball comparison diverged before action")
                r.update(id=c["id"],first_action_diff=d,stock_score=old["score"]["learner"],
                         basis=actor.additions,exact_offer_leg_changes=found)
                if any(t["learned_landed"] for t in found):
                    r["strict_cad_witness"]={tag:replay(ready,c,value) for tag,value in (("stock",old),("learned",r))}
                result["rows"].append(r)
                result["complete"]=len(result["rows"])==len(cards)
                _write(out,result)
                print(c["id"],old["score"]["learner"],r["score"]["learner"],len(actor.additions),
                      sum(k["issued"] is not None for k in actor.additions),d,found,flush=True)
        finally:
            actor.close();lens.close();a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","run"))
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args()
    schedule(a.plan) if a.mode=="plan" else run(a.plan,a.out)
