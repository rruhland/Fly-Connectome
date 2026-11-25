"""Build a signed early preparation map only from physically rewarded motion logs."""

import argparse
from pathlib import Path

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash,_read,_write
from .tt19_view_full_game import digest_json
from .tt40_low_feasibility import replay_prefix
from .tt44_paired_credit import sources,ROOT as PAIRS
from .tt49_terminal_blind import ROOT as BLIND,PROOF,PLAN
from .tt44_paired_credit import State
from .tt48_terminal_repair import ARCHIVE,terminal

ROOT=Path("artifacts/tt50_conditional_prep")
TRAIN=ROOT/"motor.json"
P_SCALE=np.array([.20,.082,.59,.75,.84])
Q_SCALE=np.array([.72,.49,.54,.52,.62])
TARGET_CUE=-.43


def incoming(offer):
    p=offer.get("receive_line")
    if p is None:
        return None
    return np.r_[p["xyz"][0],p["xyz"][2],p["velocity"]]


def calibrate(out):
    d=_read(PAIRS/"training.json.gz")
    cones=[]
    old=sources()
    for packet in d["rows"]:
        _,src=old[packet["source"]]
        event=src["rows"][packet["index"]]["placements"]["delivered"][packet["leg"]]
        x=incoming(event)
        if x is None:continue
        # More than one teacher on the same physical ball is a single example.
        mode=next((k for k,t in packet["variants"].items() if t["landed"]),None)
        if mode is None:continue
        a=packet["variants"][mode]["sensory"]
        early=next((q for q in a if q[5]<-.9 and q[2]>.04),None)
        if early is None:continue
        chosen=min(a,key=lambda q:abs(q[2]-TARGET_CUE))
        cones.append(dict(id=packet["id"],sample="direct-causal",x=x.tolist(),q=early[7:12],
                          target=chosen[19:24],real_score=True))
    d=_read(BLIND/"training-only.json.gz")
    for r in d["rows"]:
        b=r["branches"]["stock"]
        x=incoming(b["new_offer"])
        if not b["legal"] or x is None:continue
        a=[q for q in b["observations"] if q[5]<-.9 and q[0]>=b["first_inbound_tick"]]
        if not a:continue
        first=a[0]
        chosen=min(a,key=lambda q:abs(q[2]-TARGET_CUE))
        # These archived rows carry the bounded normalized motor command,
        # rather than a separate next-reference column.
        q=np.array(chosen[13:18])+np.array(chosen[19:24])*(4.16/.8/60)
        cones.append(dict(id=r["stream"],sample="strong-stock-after-departure",x=x.tolist(),q=first[7:12],
                          target=q.tolist(),real_score=True))
    result=dict(schema="tt50-present-body-continuous-prep-1",rows=cones,fitting_plane_m=-1.1,target_at_depth_m=TARGET_CUE,
                packet_sha256={str(p):_hash(p) for p in (PAIRS/"training.json.gz",BLIND/"training-only.json.gz")})
    _write(out,result)


class ContinuousMap:
    """Local affine body target with a conservative bounded early correction."""
    def __init__(self,data):
        rows=data["rows"]
        self.p=np.array([r["x"] for r in rows])/P_SCALE
        self.q=np.array([r["q"] for r in rows])/Q_SCALE
        self.u=np.array([r["target"] for r in rows])

    def query(self,p,q,ref):
        cue=p/P_SCALE
        delta=self.p-cue
        diff=self.q-q[:5]/Q_SCALE
        d=np.sum(delta**2,axis=1)+.27*np.sum(diff**2,axis=1)
        keep=np.argsort(d)[:5]
        w=np.exp(-.55*d[keep])
        if w.max()<.15 or w.sum()<.43:
            return np.zeros(5),0.
        phi=np.c_[np.ones(len(keep)),delta[keep],diff[keep]]
        regular=np.diag([.08]+[2.4]*5+[3.6]*5)
        theta=np.linalg.solve((phi.T*w) @ phi+regular,(phi.T*w) @ (self.u[keep]-ref[:5]))
        residual=np.clip(theta[0],[-.38,-.16,-.25,-.27,-.40],[.38,.16,.25,.27,.40])
        return residual,min(1.,float(w.sum()/.95))


def extract(out):
    old=_read(ARCHIVE)
    original,stock=terminal(old)
    cert=_read(PROOF)
    source_motion=next(r for r in cert["rows"] if r["physical"]["answered"] and r["exact_native_replay"])
    trial=_read(BLIND/"coach.json.gz")
    positive=next((item for r in trial["rows"] for item in r["branches"].get("blind",{}).get("replayed",[])
                  if item["physical"]["answered"] and item["independent_exact_tail"]
                  and not item["changed_tail_full_cad"]["penetration_samples"]),None)
    stream=next((r["stream"] for r in trial["rows"] if positive in r["branches"].get("blind",{}).get("replayed",[])),None)
    sources_=[dict(name="terminal-teaching-source",body_trial=source_motion["physical"],
                 card=original["card"],split_tick=cert["search_tick"],body_cut=cert["fork_tick"])]
    if positive is not None:
        source=next(r for r in trial["rows"] if r["stream"]==stream)
        cut=source["branches"]["blind"]["first_inbound_cue"]
        sources_.append(dict(name="training-new-left-reply",body_trial=positive["physical"],
                           card=dict(original["card"],reply_seed=stream),split_tick=cut,
                           body_cut=_read(PLAN)["anchor"]))
    root=_read(PLAN)["anchor"]
    result=dict(schema="tt50-observed-windows-from-clean-physical-teachers-1",rows=[],
                archive_sha256=_hash(ARCHIVE),prior_43_demonstration_sha256=_hash(PROOF),
                training_blind_coach_sha256=_hash(BLIND/"coach.json.gz"))
    with mpc.centered(old["ready"]):
        a=mpc.arena();cam,rig,stride,lo,hi=mpc.resources(a)
        axes=a._arm_ids[1][2]
        try:
            digest=replay_prefix(a,stock,original["card"],root)
            saved=State(a)
            for entry in sources_:
                saved.restore(a)
                a.card=entry["card"]
                tape=entry["body_trial"]
                rows=[]
                eye=mpc.Vision(rig,a.control_dt,raw=True)
                eye.eye=mpc.LearnedOptics(eye.eye,course.OPTIC)
                for tick in range(root,len(tape["actions"])):
                    q=a.joints.copy();ref=a.data.ctrl[axes].copy()
                    v=mpc.observed(eye,tick,cam.capture() if tick % 2==0 else None,q)
                    a.exchange(tape["actions"][tick],mpc.FAR_COMMAND)
                    rows.append(dict(tick=tick,x=None if v is None else v.tolist(),q=q.tolist(),
                                     ref=ref.tolist(),next_ref=a.data.ctrl[axes].tolist()))
                if digest_json(a.events)!=digest_json(tape["events"]):
                    raise ValueError("Optical readout diverged from previously verified teacher tape")
                p=tape["placements"]["delivered"][42]
                ni=next(e for e in tape["events"] if e.get("collider") in ("table","table_right")
                        and p["near_bounce"]["time"]<e["time"]<p["near_bounce"]["time"]+.05)
                strong=incoming(p)
                result["rows"].append(dict(name=entry["name"],mode_start=entry["split_tick"],
                    ready_change_begins=entry["body_cut"],reward="physical legal and answered",
                    x_at_receive_line=None if strong is None else strong.tolist(),
                    table_release_y=ni["ball"][1], native_touch=p["near_contact"]["location"], observations=rows,
                    event_sha256=digest_json(a.events),prefix_sha256=digest))
                _write(out,result)
                print(entry["name"],len(rows),flush=True)
        finally:
            cam.close();a.close()
    result["complete"]=True
    _write(out,result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("map","extract"))
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    ROOT.mkdir(exist_ok=True,parents=True)
    if a.mode=="map":
        calibrate(a.output or TRAIN)
    else:
        extract(a.output or ROOT/"optical-skill.json.gz")
