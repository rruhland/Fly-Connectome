"""Carry the same online credit policy into matched points with alternate private RGB sync."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt39_carry_joints import Carried
from .tt41_early_optics_carry import PLAN, PRACTICE, resume, save
from .tt42_camera_precision import LIVE_ARCHIVE, OPTIC, OUT, Shutter, installed, zoomed


def match(before, after):
    k=next((i for i,(a,b) in enumerate(zip(before["actions"],after["actions"])) if a!=b),None)
    if k is None and len(before["actions"])!=len(after["actions"]):
        k=min(len(before["actions"]),len(after["actions"]))
    last=len(after["actions"]) if k is None else k
    ep=lambda row:[e for e in row["events"] if e["step"]<=last]
    return dict(first_changed_action_tick=k,exact_physics_prefix=digest_json(ep(before))==digest_json(ep(after)),
                exact_carry_start=before["memory"]["before"]==after["memory"]["before"],
                legal_near=[before["score"]["learner"],after["score"]["learner"]],
                alternating=[before["score"]["returns"],after["score"]["returns"]])


def run(args):
    a0=_read(LIVE_ARCHIVE)
    fit=_read(args.sensor)
    how=fit["selection"]["mode"]
    ordinary_complete=bool(a0.get("complete",len(a0["rows"])==len(a0["cards"])))
    if not ordinary_complete or not fit["practice"]["complete"]:
        raise ValueError("Requires the completed, frozen practice diagnostics and control")
    a_cards=a0["cards"]
    cards=_read(PLAN)["replication"][:len(a_cards)]
    if (cards!=a_cards or _hash(PLAN)!=a0["plan_sha256"] or _hash(PRACTICE)!=a0["previous_practice"][PRACTICE.name]
            or _hash(mpc.BANK)!=a0["source_sha256"] or _hash(OPTIC)!=a0["optical_sha256"]
            or a0["mode"]!="live" or a0["variant"]!="elastic" or a0["cap"]!=200):
        raise ValueError("Baseline credit-carry configuration differs")
    past={c["id"] for c in cards}
    for role in ("practice","held"):
        if fit[role] is not None and set(fit[role]["groups"])&past:
            raise ValueError("Sensor calibration or evaluation consumed a prospective live release")
    ready=_read(mpc.BANK.with_suffix(".json"))["ready"]
    if ready!=a0["ready"]:
        raise ValueError("Original ready pose differs")
    cards=cards[:args.limit]
    with np.load(mpc.BANK) as prior:
        raw=mpc.DeepField(prior,"local")
    launches=mpc.measured_launches(args.feeds)
    checkpoint=args.output.with_suffix("").with_suffix(".npz")
    cp0=checkpoint.with_name(checkpoint.stem+"-fresh-stock.npz")
    doc=dict(schema="precision-live-1",cards=cards, sensor_variant=how, sensor_report_sha256=_hash(args.sensor),
             stock_sha256=_hash(LIVE_ARCHIVE),optic_sha256=_hash(OPTIC),ready=ready,
             config={k:a0[k] for k in ("mode","deck","variant","budget","cap",
                       "source_sha256","plan_sha256","previous_practice","credit_method")}
                    | {"capacity":a0.get("capacity",768)},
             pass_boundary="Stock vision, estimated bounce, field, credit and motor schedule always unchanged. "
             "A supplementary same-pose 768×576/60Hz stream may alter only private ball p/v at sync.",
             source_code={n:_hash(Path(__file__).with_name(n)) for n in ("tt42_camera_precision.py",
                         "tt38_joint_mpc.py","tt37_optical_clinic.py","tt29_rally_rollout.cpp")},
             control_scope="fresh first game, archived subsequent unless fresh first diverges",
             rows=[],complete=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.resume:
        existing=_read(args.output)
        if any(doc[k]!=existing[k] for k in doc.keys()-{"rows","complete"}):
            raise ValueError("Cannot resume changed prospective treatment, roster or implementation")
        doc=existing
    with mpc.centered(ready):
        a=mpc.arena()
        camera,rig,stride,lower,upper=mpc.resources(a)
        high,hi_rig=zoomed(a)
        with patch.object(mpc,"Fragments",lambda file,engine:Carried(file,engine,practice=[PRACTICE],held=past)):
            control=mpc.MPC(mpc.BANK,raw,variant=a0["variant"],budget=a0["budget"])
            treatment=mpc.MPC(mpc.BANK,raw,variant=a0["variant"],budget=a0["budget"])
        for model in (control,treatment):
            if model.priors.digest()!=a0["rows"][0]["memory"]["before"] or model.priors.online.maxlen != 768:
                raise ValueError("Initial carried credit or memory capacity differs")
        readout=Shutter(high,hi_rig,variant=how)
        installed(treatment.engine,readout,variant=how)
        def game(c,model):
            return mpc.play(a,camera,rig,stride,lower,upper,ready,c,launches[c["id"]],raw,model,
                            OPTIC,a0["mode"],a0["cap"],args.progress)
        if args.resume and doc["rows"]:
            resume(treatment,checkpoint,doc)
            if not doc.get("archived_control_verified",False):
                resume(control,cp0,{"rows":[r["control_fresh"] for r in doc["rows"]],
                                    "checkpoint_sha256":doc["control_checkpoint_sha256"]})
        try:
            for i in range(len(doc["rows"]),len(cards)):
                c=cards[i]
                before=a0["rows"][i]
                fresh=None
                if i==0 or not doc.get("archived_control_verified",False):
                    fresh=game(c,control)
                    scores=("actions","events","placements","score","memory")
                    if i==0:
                        doc["archived_control_verified"]=all(digest_json(fresh[k])==digest_json(before[k])
                                                              for k in scores)
                    if not doc["archived_control_verified"]:
                        before=fresh
                readout.begin()
                after=game(c,treatment)
                record=dict(card=c,comparison=match(before,after),precision=after,
                            add_camera=dict(readout.stats),overlays=readout.notes,control_fresh=fresh)
                if not record["comparison"]["exact_physics_prefix"]:
                    raise ValueError("Compared games differ before the first altered issued command")
                doc["rows"].append(record)
                doc["checkpoint_sha256"]=save(treatment,checkpoint)
                if not doc["archived_control_verified"]:
                    doc["control_checkpoint_sha256"]=save(control,cp0)
                doc["complete"]=len(doc["rows"])==len(cards)
                _write(args.output,doc)
                print(i+1,c["id"],record["comparison"],dict(readout.stats),flush=True)
        finally:
            treatment.close();control.close();high.close();camera.close();a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=OUT/"live.json.gz")
    p.add_argument("--sensor",type=Path,default=OUT/"accuracy.json")
    p.add_argument("--feeds",type=Path,default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--limit",type=int,default=6)
    p.add_argument("--progress",type=int,default=5)
    p.add_argument("--resume",action="store_true")
    run(p.parse_args())
