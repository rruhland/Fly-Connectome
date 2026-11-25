"""Check real already missed ordinary balls with unchanged full stock preparation until recovery."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt56_postures as h
from . import tt47_season_retention as provenance
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_opponent import draw
from .tt39_carry_joints import Carried
from .tt39_opportunity_map import projected_height
from .tt48_contact_patch_report import replay

PARENT=h.season.ROOT/"learning-96.json.gz"
STOCK=h.season.ROOT/"after96.json.gz"
PLAN=h.ROOT/"ordinary.protocol.json"
OUT=h.ROOT/"ordinary.json.gz"


class FinalRecovery(h.Recovery):
    def step(self,t,eye,x,q,ref,base):
        mode=self.kind
        if t<self.allow_after:
            self.kind="stock"
        try:
            return super().step(t,eye,x,q,ref,base)
        finally:
            self.kind=mode


def plan(out):
    a=_read(STOCK)
    prior={r["id"] for r in _read(h.MODEL)["rows"]}
    records=[]
    for i,(c,r) in enumerate(zip(a["cards"],a["rows"])):
        if not 1<=r["score"]["learner"]<=3 or c["id"] in prior:
            continue
        last=r["placements"]["delivered"][-1]
        if last["near_landed"] or last["receive_line"] is None:
            continue
        measure=projected_height(last["receive_line"])
        if measure<1.025:
            continue
        held_before=last["far_contact"]["time"]
        sep=max(e["time"] for e in r["events"] if e["kind"]=="separation" and
                e.get("collider")=="learner_racket" and e["time"]<held_before)
        gate=int(np.ceil(sep/h.mpc.DT-1e-8))
        records.append(dict(index=i,card=c,arm_returns=r["score"]["learner"],gate=gate,leg=last["leg"],grade_height=measure))
        if len(records)==2:
            break
    if len(records)!=2 or a["carry_sha256"]!=_hash(PARENT):
        raise ValueError("No complete separate ordinary-failure holdout")
    sources=[STOCK,PARENT,h.MODEL,Path(__file__).relative_to(Path.cwd()),Path(h.__file__).relative_to(Path.cwd())]
    _write(out,dict(records=records,sources={str(p):_hash(p) for p in sources},
                    apply="Only the evaluator's stage is pinned using the previous already completed learner separation. The controller must still receive its own consecutive valid RGB outgoing cues before issuing the same broad posture. Select first two official completed indexed streams with 1–3 previous returns and existing terminal misses above 1.025 m, disjoint from encoder training."))


def run(out,p):
    design,stock,carry,poses=map(_read,(p,STOCK,PARENT,h.MODEL))
    if any(_hash(Path(p))!=s for p,s in design["sources"].items()):
        raise ValueError("Changed ordinary-offer docket")
    ans=dict(protocol_sha256=_hash(p),rows=[],complete=False)
    if out.exists():
        old=_read(out)
        if ans["protocol_sha256"]!=old["protocol_sha256"]:
            raise ValueError("Wrong ordinary offer continuation")
        ans=old
    with np.load(h.mpc.BANK) as f:
        raw=h.mpc.DeepField(f,"local")
    with h.mpc.centered(stock["ready"]),h.enabled():
        a=h.mpc.arena()
        cad=h.mpc.arena(detail="cad",width=64,height=48)
        guard=h.PoseClearance(cad)
        cam,rig,stride,lo,hi=h.mpc.resources(a)
        ids={r["card"]["id"] for r in design["records"]}
        with patch.object(h.mpc,"Fragments",lambda b,e: Carried(b,e,practice=[h.course.PRACTICE],held=ids)):
            actor=FinalRecovery(h.mpc.BANK,raw,coverage=h.fresh.shot.Coverage(_read(h.fresh.RISK)),
                                variant=stock["variant"],budget=stock["budget"],guard=guard,
                                postures=poses["portfolio"],kind="stock")
        try:
            provenance.ancestral_restore(actor,PARENT.with_suffix("").with_suffix(".npz"),carry,h.season.CARRY)
            for box in design["records"][len(ans["rows"]):]:
                i=box["index"]; c=box["card"]; old=stock["rows"][i]
                modes={}
                actor.allow_after=box["gate"]
                for mode in ("hub","ports"):
                    actor.kind=mode
                    result=h.mpc.play(a,cam,rig,stride,lo,hi,stock["ready"],c,old["shot"],raw,actor,h.course.OPTIC,
                                      "frozen",2*(box["arm_returns"]+1))
                    check=h.compare(old,result)
                    if result["actions"][:box["gate"]]!=old["actions"][:box["gate"]]:
                        raise ValueError("Treatment changed the exact stock prefix before recovery")
                    if any(v["target"]!=list(draw(c,j,"random",None)[0])
                           for j,v in enumerate(result["placements"]["requested"])):
                        raise ValueError("Altered broad reactive target")
                    result.update(check,posture_routes=actor.passes)
                    success=next((v["learned"] and not v["stock"] for v in check["matched"] if v["leg"]==box["leg"]),False)
                    result["same_ball_rescue"]=success
                    if success:
                        result["strict_cad"]={k:replay(stock["ready"],c,v) for k,v in (("stock",old),(mode,result))}
                    modes[mode]=result
                ans["rows"].append(dict(test=box,arms=modes))
                ans["complete"]=len(ans["rows"])==len(design["records"])
                _write(out,ans)
                print(c["id"],box["grade_height"],{k:(r["same_ball_rescue"],r["first_action_diff"],
                      r["posture_routes"],r["matched"][-1:]) for k,r in modes.items()},flush=True)
        finally:
            actor.close();cam.close();cad.close();a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","run"))
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args()
    plan(a.plan) if a.mode=="plan" else run(a.out,a.plan)
