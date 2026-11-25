"""Matched diagnostic on offer 43 of the 42-return point, keeping its exact earned bank."""

import argparse
from pathlib import Path
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt39_carry_joints import Carried
from .tt39_opportunity_map import projected_height
from .tt44_continuous_credit import SEEDS
from .tt44_continuous_credit_fit import Local
from .tt44_paired_credit import State, prefix, suffix
from .tt48_exact_repair import MODEL, ROOT, Repair

ARCHIVE=season.ROOT/"after-two.json.gz"
VERIFY=season.ROOT/"random-42.json"
BANK=ARCHIVE.with_suffix("").with_suffix(".npz")
DEFAULTS=("stock-reset","learned","9519","4283","9519-extended","4283-extended")


def terminal(d):
    r,c=d["rows"][1],d["cards"][1]
    o=r["placements"]["delivered"][-1]
    leg=o["leg"]
    t=next(e["time"] for e in r["events"] if e.get("collider")=="rival_racket"
           and o["far_contact"]["time"]<=e["time"]<o["near_bounce"]["time"])
    audit=_read(VERIFY)
    if (leg!=42 or r["score"]["learner"]!=42 or c["id"]!=2741 or c!=audit["card"]
            or digest_json(r["events"])!=audit["event_sha256"] or o["near_landed"] 
            or any(e["kind"]=="legal_return" and e["side"]==1 and e["time"]>t for e in r["events"])):
        raise ValueError("Not the original terminal challenge with the complete final bank")
    return dict(id=c["id"],card=c,leg=leg,tick=int(np.ceil(t/mpc.DT-1e-7)),
                projected_m=projected_height(o["receive_line"]),original_offer=o),r


def clearance_sample(a):
    blade=a._racket_ids[1]
    axis=a.data.geom_xmat[blade].reshape(3,3)[:,2]
    center=a.data.geom_xpos[blade].copy()
    d=a.ball-center
    axial=float(abs(d@axis))
    tangent=float(np.linalg.norm(d-(d@axis)*axis))
    radius,thickness=a.model.geom_size[blade,:2]
    ball_radius=a.model.geom_size[a._geom_ids["ball"],0]
    gap=float(np.hypot(max(tangent-radius,0.),max(axial-thickness,0.))-ball_radius)
    return dict(clearance_m=gap,time_s=float(a.data.time),ball_xyz=a.ball.tolist(),blade_xyz=center.tolist(),
                axial_m=axial,tangential_m=tangent,joint_rad=a.joints.tolist())


def inspect_original(a,state,old,case):
    state.restore(a)
    hits=[]
    for cmd in old["actions"][case["tick"]:]:
        if a.velocity[1]<-.9 and -1.83<a.ball[1]<-1.03:
            hits.append(clearance_sample(a))
        a.exchange(cmd,mpc.FAR_COMMAND)
    if digest_json(a.events)!=digest_json(old["events"]):
        raise ValueError("Physical restore did not reproduce the exact terminal miss")
    return min(hits,key=lambda p:p["clearance_m"]) if hits else None


def run(out,modes):
    d=_read(ARCHIVE)
    case,old=terminal(d)
    mo=_read(MODEL)
    model=Local(mo["rows"],[r["context"] for r in mo["easy"]])
    demos={int(r["id"]):next(np.array(t["u"]) for t in r["trials"] if t["reward"]>r["stock"])
           for r in mo["rows"] if r["id"] in (9519,4283)}
    examples=[(np.array(r["context"]),demos[r["id"]],r["id"]) for r in mo["rows"] if r["id"] in demos]
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    boot=np.asarray(d["ready"])
    result=dict(schema="tt48-fixed-record-terminal-1",source_sha256=_hash(ARCHIVE),tape_audit_sha256=_hash(VERIFY),
                final_bank_sha256=_hash(BANK),learner_sha256=_hash(MODEL),case=case,modes=list(modes),rows=[],
                boundary="Exact original physical offer and 42-return memory. Stock reset does not claim the original planner's RNG/warm cache.")
    with mpc.centered(boot):
        a=mpc.arena()
        camera,rig,stride,lo,hi=mpc.resources(a)
        with patch.object(mpc,"Fragments",lambda f,e:Carried(f,e,practice=[course.PRACTICE],held={case["id"]})):
            actor=Repair(mpc.BANK,raw,readout=model,examples=examples,coverage=fresh.shot.Coverage(_read(fresh.RISK)),
                         budget=d["budget"],variant=d["variant"])
        try:
            carried=_read(season.CARRY)
            # The checkpoint serial includes earlier continuous practice,
            # while this receipt stores the two additional native points.
            doc={**d,"rows":[None]*len(carried["rows"])+d["rows"]}
            course.resume(actor,BANK,doc)
            start=process_time()
            data=prefix(a,case["card"],old,case["tick"],camera,a._arm_ids[1][2])
            saved=State(a)
            result["original_sampled_gap"]=inspect_original(a,saved,old,case)
            result["prefix_cpu_s"]=process_time()-start
            result["original_actions_sha256"]=digest_json(old["actions"][:case["tick"]])
            result["memory_at_fork"]=actor.priors.digest()
            _write(out,result)
            print("exact offer",case["leg"]+1,"projected height",case["projected_m"],"archived gap",
                  result["original_sampled_gap"],flush=True)
            for mode in modes:
                saved.restore(a)
                actor.keep_until=-.91 if mode.endswith("-extended") else -.53
                actor.mode=mode if mode=="learned" else "stock"
                actor.forced=None
                if mode.startswith(("9519","4283")):
                    actor.mode="forced"
                    actor.forced=demos[int(mode.split("-")[0])]
                elif mode.startswith("spread-"):
                    actor.mode="forced"
                    actor.forced=SEEDS[int(mode.split("-")[1])]
                run=suffix(a,camera,rig,stride,lo,hi,actor,data,case,old,boot)
                run.update(mode_label=mode,choices=actor.picked,actual_own=42+run["landed"],
                           same_archived_commands=run["controls"]==old["actions"][case["tick"]:],
                           archived_prefix_solver_exact=True)
                result["rows"].append(run)
                _write(out,result)
                print(mode,run["landed"],run["touched"],
                      [c["parameters"] for c in run["choices"]],flush=True)
        finally:
            actor.close();camera.close();a.close()
    result["complete"]=True
    _write(out,result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"terminal-43.json.gz")
    p.add_argument("--modes",nargs="+",default=DEFAULTS,
                   choices=(*DEFAULTS,*(f"spread-{i}" for i in range(len(SEEDS)))))
    a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    run(a.output,a.modes)
