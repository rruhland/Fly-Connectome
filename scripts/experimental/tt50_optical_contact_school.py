"""Check an RGB stroke independently of the precontact classifier at a taught native posture."""

import argparse
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from . import tt50_conditional_prep as arm
from .tt37_opponent_audit import CourtClearance, guarded
from .tt39_opportunity_map import projected_height


def inputs():
    proof=arm._read(arm.PROOF)
    example=next(r for r in proof["rows"] if r["physical"]["answered"] and r["exact_native_replay"])
    a=dict(name="terminal-teaching-source",cut=proof["search_tick"],tape=example["physical"])
    d=arm._read(arm.BLIND/"coach.json.gz")
    for row in d["rows"]:
        b=row["branches"]["blind"]
        p=next((n for n in b["replayed"] if n["physical"]["answered"] and n["independent_exact_tail"]),None)
        if p is not None:
            return example["port"],(a,dict(name="training-new-left-reply",stream=row["stream"],
                                           cut=b["first_inbound_cue"],tape=p["physical"]))
    raise ValueError("Need a score-earned disjoint training motion")


def run(output):
    src=arm._read(arm.ARCHIVE)
    case,archive=arm.terminal(src)
    p,offers=inputs()
    planner=arm._read(arm.PLAN)
    d=arm._read(arm.SKILL)
    policy=arm._read(arm.DIRECT_MODEL)
    local=arm.Local(policy["rows"],[z["context"] for z in policy["easy"]])
    motor=arm.ContinuousMap(arm._read(arm.TRAIN))
    doc=dict(schema="tt50-optical-contact-school-1",actor_sha256=arm._hash(Path(arm.__file__)),
             skill_sha256=arm._hash(arm.SKILL),rows=[],
             role="Teaching trajectories only. Original realized body route is replayed up to the indicated native boundary. All subsequent choices are RGB and encoder based.")
    with np.load(arm.mpc.BANK) as z:
        raw=arm.mpc.DeepField(z,"local")
    with arm.mpc.centered(src["ready"]):
        a=arm.mpc.arena();cad=arm.mpc.arena(detail="cad")
        contact=CourtClearance(cad.model,articulated=True)
        res=arm.mpc.resources(a)
        camera,rig,stride,lo,hi=res
        axes=a._arm_ids[1][2]
        checker=arm.Reach(cad,src["ready"],src["ready"])
        with patch.object(arm.mpc,"Fragments",lambda f,e:arm.Carried(f,e,practice=[arm.course.PRACTICE],held={case["id"]})):
            actor=arm.Arm(arm.mpc.BANK,raw,readout=local,examples=(),
                coverage=arm.fresh.shot.Coverage(arm._read(arm.fresh.RISK)),budget=src["budget"],variant=src["variant"],
                predictor_path=arm.CAMERA,affine=motor,taught=arm.Taught(d),checker=checker,
                waypoint=p["waypoint"],approach=p["approach"])
        try:
            past={**src,"rows":[None]*len(arm._read(arm.season.CARRY)["rows"])+src["rows"]}
            arm.course.resume(actor,arm.BANK,past)
            root=planner["anchor"]
            history=arm.prefix(a,case["card"],archive,root,camera,axes,window=32)
            saved=arm.State(a)
            cold=arm.Observer(a,rig,actor,history)
            for t in offers:
                saved.restore(a)
                stream=t.get("stream",case["card"]["reply_seed"])
                a.card=dict(case["card"],reply_seed=stream)
                eye=cold.restore(actor)
                outgoing=None;inbound=None
                tape=t["tape"]
                for tick in range(root,t["cut"]):
                    q=a.joints.copy();ref=a.data.ctrl[axes].copy()
                    f=camera.capture() if tick%2==0 else None
                    x=arm.mpc.observed(eye,tick,f,q)
                    actor.engine.observe(tick,eye.eye,q,ref)
                    if x is not None and x[4]>.9:outgoing=tick
                    if outgoing is not None and x is not None and x[4]<-.9 and inbound is None and tick-outgoing<40:
                        inbound=tick
                    a.exchange(tape["actions"][tick],arm.mpc.FAR_COMMAND)
                state=arm.State(a)
                # Native installation and optics are both common. Snapshot
                # them before any actor gets to issue a contact action.
                eye0=deepcopy(eye)
                e=actor.engine
                body=mujoco.MjData(e.body_model)
                mujoco.mj_copyData(body,e.body_model,e.body_state)
                keys=("previous","rate","tick","reference","bounced")
                engine0={k:deepcopy(getattr(e,k)) for k in keys}
                event0=arm.digest_json(a.events)
                q0=a.joints.copy();u0=a.data.ctrl[axes].copy()
                observed=next(j for z in d["rows"] if z["name"]==t["name"]
                              for j in z["observations"] if j["tick"]==t["cut"])
                if not np.array_equal(q0,observed["q"]) or not np.array_equal(u0,observed["ref"]):
                    raise ValueError("Changed the score-earned body or motor reference at handoff")
                source=tape["placements"]["delivered"][case["leg"]]
                row=dict(stream=stream,name=t["name"],cut=t["cut"],prefix_event_sha256=event0,
                         true_incoming_height=projected_height(source["receive_line"]),
                         q=q0.tolist(),ref=u0.tolist(),variants={})
                for mode in ("stock","body-learned"):
                    state.restore(a)
                    guarded(a,contact)
                    actor.begin();actor.policy_mode=mode
                    actor.y_out=outgoing;actor.seen_inbound_at=inbound
                    eye=deepcopy(eye0)
                    mujoco.mj_copyData(e.body_state,e.body_model,body)
                    for k,v in engine0.items():setattr(e,k,deepcopy(v))
                    e.tracker=eye.eye;e.latest=eye.eye.latest
                    if mode=="body-learned":
                        actor.hard_active=True;actor.plan_t=t["cut"]
                        actor.plan=np.repeat(u0[None,:],20,axis=0)
                    commands=[];sensed=[]
                    for tick in range(t["cut"],t["cut"]+140):
                        q=a.joints.copy();ref=a.data.ctrl[axes].copy()
                        f=camera.capture() if tick%2==0 else None
                        x=arm.mpc.observed(eye,tick,f,q)
                        g=actor.blended(tick,f,eye.eye,x,q,ref,actor.priors.home)
                        u=np.clip((np.clip(g,lo,hi)-ref)/stride,-.8,.8).tolist()
                        commands.append(u)
                        if x is not None:sensed.append([tick,*x])
                        _,_,done,capped,_=a.exchange(u,arm.mpc.FAR_COMMAND)
                        if done or capped or a.rival_returns>=44:break
                    obtained=a.point_placements()["delivered"][case["leg"]]
                    exact=all(arm.same_trace(obtained[k],source[k]) for k in ("far_contact","near_bounce"))
                    if not exact:raise ValueError("Changed the supervised native incoming ball")
                    row["variants"][mode]=dict(own=a.legal_returns-1,answered=a.rival_returns>=44,legal=obtained["near_landed"],
                       actual_offer_exact=exact,touch=obtained["near_contact"],commands=commands,sensory=sensed,
                       trial=actor.shots,clearance=a.clearance_result(),event_sha256=arm.digest_json(a.events),
                       terminal_reason=next((v.get("reason") for v in a.events[::-1] if v["kind"]=="point"),None))
                doc["rows"].append(row)
                arm._write(output,doc)
                print(t["name"],[(m,v["legal"],v["answered"],len(v["trial"])) for m,v in row["variants"].items()],flush=True)
        finally:
            actor.close();camera.close();a.close();cad.close()
    doc["complete"]=True
    arm._write(output,doc)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=arm.ROOT/"contact-school.json.gz")
    run(p.parse_args().output)
