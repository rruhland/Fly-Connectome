"""Replace only the camera readout in the archived 12-game carry comparison."""
import argparse
from ast import literal_eval
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc, tt41_early_optics as optical
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_visual_inertia import LearnedOptics
from .tt39_carry_joints import Carried

ROOT=Path("artifacts")
CONTROL=ROOT/"tt39/credit-carry.json.gz"
PRACTICE=ROOT/"tt37/camera-time-school.npz"
PLAN=ROOT/"tt37/memory-course.json"
CAPACITY=768


def save(model,path):
    bank=model.priors.online
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream,states=np.asarray([r[0] for r in bank]),
                           goals=np.asarray([r[1] for r in bank]),
                           tags=np.array([repr(r[2]) for r in bank]),serial=model.serial)
    digest=_hash(tmp)
    tmp.replace(path)
    return digest


def resume(model,path,document):
    with np.load(path) as f:
        model.priors.online.clear()
        model.priors.online.extend(zip(f["states"],f["goals"],map(literal_eval,f["tags"]),strict=True))
        model.serial=int(f["serial"])
    if (model.serial!=len(document["rows"]) or model.priors.digest()!=document["rows"][-1]["memory"]["after"]
            or _hash(path)!=document["checkpoint_sha256"]):
        raise RuntimeError("Point-boundary checkpoint does not match archived continuation")


def run(args):
    baseline=_read(CONTROL)
    cards=_read(PLAN)["replication"][:len(baseline["cards"])]
    held={c["id"] for c in cards}
    if (cards!=baseline["cards"] or _hash(PLAN)!=baseline["plan_sha256"]
            or _hash(PRACTICE)!=baseline["previous_practice"][PRACTICE.name]
            or _hash(mpc.BANK)!=baseline["source_sha256"]
            or _hash(optical.CHECKPOINT)!=baseline["optical_sha256"]
            or baseline["mode"]!="live" or baseline["cap"]!=200):
        raise ValueError("Original credit-carry configuration is not reproduced")
    ready=_read(mpc.BANK.with_suffix(".json"))["ready"]
    if ready!=baseline["ready"]:
        raise ValueError("Ready pose changed")
    with np.load(mpc.BANK) as bank:
        raw=mpc.DeepField(bank,"local")
    launch=mpc.measured_launches(args.feeds)
    receipt=dict(mode=baseline["mode"],deck=baseline["deck"],cards=cards,ready=ready,
                 cap=baseline["cap"],budget=baseline["budget"],variant=baseline["variant"],
                 source_sha256=baseline["source_sha256"],plan_sha256=baseline["plan_sha256"],
                 previous_practice=baseline["previous_practice"],optical_sha256=baseline["optical_sha256"],
                 estimator_sha256=_hash(Path(optical.__file__)),baseline_sha256=_hash(CONTROL),
                 capacity=CAPACITY,model_source_sha256={
                     name:_hash(Path(mpc.__file__).with_name(name))
                     for name in ("tt38_joint_mpc.py","tt37_optical_clinic.py","tt29_rally_rollout.cpp")},
                 credit_method=baseline["credit_method"],
                 optical_boundary="Causal table phase and initial p/v only; stock readout after agreement",
                 rows=[])
    checkpoint=args.output.with_suffix("").with_suffix(".npz")
    if args.resume:
        document=_read(args.output)
        for k in receipt.keys()-{"rows"}:
            if document[k]!=receipt[k]:
                raise ValueError(f"Attempt to resume with changed {k}")
    else:
        document=receipt
    _write(args.output,document)
    with mpc.centered(ready):
        a=mpc.arena()
        camera,rig,stride,lo,hi=mpc.resources(a)
        with patch.object(mpc,"Fragments",lambda file,engine: Carried(file,engine,practice=[PRACTICE],held=held)):
            model=mpc.MPC(mpc.BANK,raw,variant=baseline["variant"],budget=baseline["budget"])
        initial=model.priors.digest()
        if initial!=baseline["rows"][0]["memory"]["before"] or model.priors.online.maxlen!=CAPACITY:
            raise ValueError("Initial carried motor memory differs from control")
        if args.resume:
            resume(model,checkpoint,document)
        try:
            for i in range(len(document["rows"]),len(cards)):
                c=cards[i]
                monitors=[]
                def factory(tracker,point):
                    readout=optical.EarlyReadout(point)
                    monitors.append(readout)
                    return LearnedOptics(tracker,readout)
                with patch.object(mpc,"LearnedOptics",factory):
                    row=mpc.play(a,camera,rig,stride,lo,hi,ready,c,launch[c["id"]],raw,
                                 model,optical.CHECKPOINT,baseline["mode"],baseline["cap"],args.progress)
                row["new_optical"]=[n for m in monitors for n in m.causal_notes]
                row["inserted_optical_cpu_s"]=sum(m.additional_cpu for m in monitors)
                old=baseline["rows"][i]
                k=next((k for k,(l,r) in enumerate(zip(old["actions"],row["actions"])) if l!=r),None)
                row["first_action_diff_from_archived_stock"]=k
                row["same_starting_memory"]=row["memory"]["before"]==old["memory"]["before"]
                t=len(row["actions"]) if k is None else k
                row["exact_physical_prefix"] = (
                    digest_json([e for e in old["events"] if e["step"]<=t]) ==
                    digest_json([e for e in row["events"] if e["step"]<=t]))
                document["rows"].append(row)
                document["checkpoint_sha256"]=save(model,checkpoint)
                document["complete"]=len(document["rows"])==len(cards)
                _write(args.output,document)
                print("paired",i+1,c["id"],old["score"]["learner"],row["score"]["learner"],k,
                      row["memory"]["after"]["rows"],round(row["inserted_optical_cpu_s"],4),flush=True)
        finally:
            model.close();camera.close();a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--feeds",type=Path,default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--resume",action="store_true")
    p.add_argument("--progress",type=int,default=10)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    run(a)
