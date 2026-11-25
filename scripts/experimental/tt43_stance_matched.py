"""Give the camera-driven hub actor control at two selected, matched low-reply prefixes."""

import argparse
from pathlib import Path

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt33_joint import FAR_COMMAND, Vision
from .tt37_opponent import draw
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded, verify_clearance
from .tt40_long_low import same_trace
from .tt40_low_feasibility import SOURCE
from .tt43_stance import OUT, ROOT
from .tt43_stance_live import OneHub, load_memories
from .tt43_stance_native import fork_ticks
from .tt43_stance_uncertain import LowHub, fitted_camera_examples

CASES = ((4322,5),(8302,3),(14843,1))


def verify(output, chart_path, witness_path, cases, ensemble=False, teacher=None, optical_teacher=None, warm=False):
    old=_read(SOURCE)
    chart=_read(chart_path)
    boot=np.asarray(old["ready"])
    hub=np.r_[chart["routes"]["either_far"][1]["q"],boot[-1]]
    w=_read(witness_path)
    memory=load_memories(teacher,{i for i,_ in cases})
    if optical_teacher is not None:
        opt=_read(optical_teacher)
        proof=optical_teacher.with_name("same-offer-diversity.json.gz")
        verify_source=_read(proof)
        if (_hash(proof)!=opt["source_sha256"] or verify_source["card"]["id"] in {i for i,_ in cases}
                or not opt["sensory_native_truth_excluded"] or not verify_source["complete"]):
            raise ValueError("Use the release-disjoint completed optical proof")
    atlas_path=ROOT/"tt41/shallow-atlas.json.gz"
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    d=dict(schema="tt43-optical-low-matched-forks-1",rows=[],boot=boot.tolist(),hub=hub.tolist(),
           chart_sha256=_hash(chart_path),witness_sha256=_hash(witness_path),source_sha256=_hash(SOURCE),
           atlas_sha256=_hash(atlas_path),optics_sha256=_hash(course.OPTIC),
           teacher_sha256=None if teacher is None else _hash(teacher),teacher_examples=len(memory),ensemble=ensemble,warm=warm,
           scope="Scheduled prior-physical action prefix only. Subsequent choices use RGB/encoders. Optional teacher is release disjoint.")
    if optical_teacher is not None:
        payload=optical_teacher.with_suffix("").with_suffix(".npz")
        if _hash(payload)!=opt["motion_sha256"]:
            raise ValueError("Optical example tape does not match its receipt")
        d["optical_teacher_sha256"]=_hash(optical_teacher)
    with mpc.centered(boot):
        a=mpc.arena()
        cad=mpc.arena(detail="cad")
        camera,rig,stride,lo,hi=mpc.resources(a)
        actor=(LowHub(mpc.BANK,raw,hub,_read(atlas_path),memory,handoff=True,warm=warm) if ensemble else
               OneHub(mpc.BANK,raw,hub,_read(atlas_path),memory,handoff=True))
        if optical_teacher is not None:
            with np.load(payload) as z:
                exp=fitted_camera_examples(actor.chart,opt["camera_response_rows"],z)
            actor.earned.extend(exp)
            d["optical_teacher_examples"]=len(exp)
        guard=CourtClearance(cad.model,articulated=True)
        try:
            for ident,leg in cases:
                i=next(i for i,c in enumerate(old["cards"]) if c["id"]==ident)
                card,orig=old["cards"][i],old["rows"][i]
                target=next(r for r in w["low"] if r["id"]==ident and r["face"]=="negative")
                fork,_=fork_ticks(orig,target,a.control_dt)
                a.launch(orig["shot"],card=card)
                a.partner.reset(seed=card["seed"])
                guarded(a,guard)
                eye=Vision(rig,a.control_dt,raw=True)
                eye.eye=mpc.LearnedOptics(eye.eye,course.OPTIC)
                actor.begin()
                actions=[]
                seen_inbound=False
                contact=[]
                with contact_observer(a,contact):
                    for tick in range(fork+285):
                        q=a.joints
                        reference=a.data.ctrl[a._arm_ids[1][2]].copy()
                        frame=camera.capture() if tick>=fork-16 and tick%2==0 else None
                        x=mpc.observed(eye,tick,frame,q) if tick>=fork-16 else None
                        if tick<fork:
                            actor.engine.observe(tick,eye.eye,q,reference.copy())
                            cmd=orig["actions"][tick]
                        else:
                            if x is not None and x[4]<-.9 and x[1]<-.90:
                                seen_inbound=True
                            base=boot.copy()
                            if x is not None and (x[4]<-.9 or (seen_inbound and x[1]<.50)):
                                wanted,support=raw.get(x,q)
                                if wanted is not None:
                                    base+=support*(wanted-boot)
                            want=actor.step(tick,eye.eye,x,q,reference,base)
                            cmd=np.clip((np.clip(want,lo,hi)-reference)/stride,-.8,.8).tolist()
                        _,_,done,capped,_=a.exchange(cmd,FAR_COMMAND)
                        actions.append(cmd)
                        if done or capped or a.rival_returns>=leg+2:
                            break
                verify_clearance(a)
                placements=a.point_placements()
                draws=[r["target"] for r in placements["requested"]]
                if any(list(draw(card,j,"random",None)[0])!=x for j,x in enumerate(draws)):
                    raise RuntimeError("Opponent draw changed")
                actual=placements["delivered"][leg] if len(placements["delivered"])>leg else None
                offered=orig["placements"]["delivered"][leg]
                common=bool(actual and all(same_trace(actual[k],offered[k]) for k in ("far_contact","near_bounce")))
                physical=[] if actual is None else [e for e in contact if e["side"]==1 and
                                                       e["time_s"]>actual["near_bounce"]["time"]]
                row=dict(id=ident,leg=leg,card=card,shot=orig["shot"],fork=fork,stock_landed=offered["near_landed"],
                         shared_delivery=common,landed=bool(actual and actual["near_landed"]),
                         answered=bool(actual and actual["near_landed"] and len(placements["delivered"])>leg+1),
                         clearance=a.clearance_result(),contacts=physical,score=audit(a.events,opening_side=1),
                         planner=dict(actor.diag),notes=actor.notes,placements=placements,events=a.events,
                         actions=actions,motor_audit=a.motor_audit())
                a.launch(orig["shot"],card=card)
                a.partner.reset(seed=card["seed"])
                for cmd in actions:
                    a.exchange(cmd,FAR_COMMAND)
                if digest_json(a.events)!=digest_json(row["events"]):
                    raise RuntimeError("Independent tape replay diverged")
                d["rows"].append(row)
                _write(output,d)
                print(ident,common,row["landed"],row["answered"],row["planner"],flush=True)
        finally:
            actor.close();camera.close();a.close();cad.close()
    d["complete"]=True
    _write(output,d)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=OUT/"camera-low.json.gz")
    p.add_argument("--chart",type=Path,default=OUT/"chart.json.gz")
    p.add_argument("--witnesses",type=Path,default=OUT/"witness-envelope.json.gz")
    p.add_argument("--limit",type=int,default=2)
    p.add_argument("--cases",nargs="+",type=int,help="zero-based fixed development cases")
    p.add_argument("--ensemble",action="store_true")
    p.add_argument("--warm",action="store_true")
    p.add_argument("--teacher",type=Path)
    p.add_argument("--optical-teacher",type=Path)
    a=p.parse_args()
    cases=CASES[:a.limit] if a.cases is None else [CASES[i] for i in a.cases]
    verify(a.output,a.chart,a.witnesses,cases,a.ensemble,a.teacher,a.optical_teacher,a.warm)
