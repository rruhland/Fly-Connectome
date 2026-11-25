"""Test the same visible-outbound preparation across fresh native reply draws."""

import argparse
from copy import deepcopy
from pathlib import Path
from time import process_time
from unittest.mock import patch

import mujoco
import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent import draw
from .tt39_carry_joints import Carried
from .tt39_opportunity_map import projected_height
from .tt40_long_low import preparation,same_trace
from .tt44_paired_credit import State,prefix,warm
from .tt48_exact_repair import MODEL, Repair
from .tt44_continuous_credit_fit import Local
from .tt48_terminal_repair import ARCHIVE,BANK,terminal

ROOT=Path("artifacts/tt49_terminal_blind")
PROOF=Path("artifacts/tt48_exact_repair/terminal-effect-aligned.json.gz")
PLAN=ROOT/"protocol.json"
DURATION=53 # 0.883 seconds by motor clock, independent of the next ball
PROTOCOL_SEED=49535177


def freeze(path):
    d=_read(ARCHIVE)
    case,old=terminal(d)
    p=_read(PROOF)
    teacher=next(r for r in p["rows"] if r["physical"]["answered"] and r["exact_native_replay"]
                 and not r["changed_tail_clearance"]["penetration_samples"])
    prior=old["placements"]["requested"][case["leg"]]
    # The target draw happened while the previous near shot was still
    # separating. Copy the last control boundary before that actual draw.
    anchor=int(np.floor(prior["time_s"]/mpc.DT+1e-7))
    rng=np.random.default_rng(PROTOCOL_SEED)
    seeds=[int(v) for v in rng.integers(1,2**30,size=6)]
    if len(set(seeds))!=len(seeds) or case["card"]["reply_seed"] in seeds:
        raise RuntimeError("Non-independent holdout seed")
    roster=[]
    for j,s in enumerate(seeds):
        card=dict(case["card"],reply_seed=s)
        a,_=draw(card,case["leg"],"random",None)
        roster.append(dict(split="training" if j<3 else "held",stream=s,goal=list(a)))
    # Source stroke is a geometric teaching example. Only its intermediate
    # and approach joint references survive, not the contact or timing tape.
    _write(path,dict(schema="tt49-terminal-blind-protocol-1",source_sha256=_hash(ARCHIVE),
        teacher_sha256=_hash(PROOF),roster=roster,leg=case["leg"],anchor=anchor,
        previous_swing_controls_through_tick=p["fork_tick"]-1,
        role="Before any outcomes, draw six new seeds from unchanged TT37 broad box at the same native outgoing state.",
        joint_teaching=dict(waypoint=teacher["port"]["waypoint"],approach=teacher["port"]["approach"],duration=DURATION),
        public_gate=dict(outgoing_vy_min=.90,outgoing_y=[-1.40,1.05],requires_current_outgoing=True,
                         close_on_first_visible_inbound=True)))


class Observer:
    """Reuse the pre-fork camera/body state across each native counterfactual."""
    def __init__(self,a,rig,actor,readings):
        eye=warm(a,rig,actor,readings)
        self.eye=deepcopy(eye)
        e=actor.engine
        self.state=mujoco.MjData(e.body_model)
        mujoco.mj_copyData(self.state,e.body_model,e.body_state)
        self.small={k:deepcopy(getattr(e,k)) for k in
                    ("previous","rate","tick","reference","bounced")}

    def restore(self,actor):
        actor.begin()
        e=actor.engine
        eye=deepcopy(self.eye)
        mujoco.mj_copyData(e.body_state,e.body_model,self.state)
        for k,v in self.small.items():setattr(e,k,deepcopy(v))
        e.tracker=eye.eye
        e.latest=eye.eye.latest
        return eye


class Blind:
    def __init__(self,teacher,mode,prior_swing_stop):
        self.mode=mode
        self.teacher=teacher
        self.ready=None
        self.motion=None
        self.last_out=None
        self.armed_tick=None
        self.inbound=None
        self.prior_swing_stop=prior_swing_stop
        self.timed_out=False

    def goal(self,tick,x,q,ref,original):
        if tick<self.prior_swing_stop or self.inbound is not None:
            return original
        if x is not None and x[4]>.9:
            self.last_out=tick
            if self.motion is None and -1.4<x[1]<1.05:
                self.armed_tick=tick
                g=self.teacher
                self.motion=preparation(ref[:5],np.asarray(g["waypoint"]),np.asarray(g["approach"]),
                                        g["duration"],ref[-1],brisk=True)
        if (self.last_out is not None and x is not None and x[4]<-.9
                and abs(x[1])<1.72 and tick-self.last_out<44):
            self.inbound=tick
            self.ready=q.tolist()
            return original
        if self.mode=="stock" or self.motion is None:
            return original
        if tick-self.armed_tick>=len(self.motion)+6:
            self.timed_out=True
            return original
        return self.motion[min(tick-self.armed_tick,len(self.motion)-1)]


def race(a,actor,eye,res,stock,case,reg,condition):
    image,rig,stride,lo,hi=res
    control_channels=a._arm_ids[1][2]
    start=reg["anchor"]
    controls=[]; sensed=[]; episode=Blind(reg["joint_teaching"],condition,
                                         reg["previous_swing_controls_through_tick"]+1)
    home=np.asarray(actor.priors.home)
    leg=case["leg"];outcome=None
    for tick in range(start,start+240):
        q=a.joints.copy()
        ref=a.data.ctrl[control_channels].copy()
        x=mpc.observed(eye,tick,image.capture() if tick%2==0 else None,q)
        if tick<episode.prior_swing_stop:
            actor.engine.observe(tick,eye.eye,q,ref)
            u=stock["actions"][tick]
        else:
            g=actor.get_base(tick,x,q,ref,home)
            want=actor.step(tick,eye.eye,x,q,ref,g)
            before=episode.inbound
            blind=episode.goal(tick,x,q,ref,want)
            if episode.inbound is not None and before is None:
                # Ordinary causal actor takes over immediately on detection.
                if a.legal_returns<43:
                    raise ValueError("The previous stroke did not score before the new reception")
            u=np.clip((np.clip(blind,lo,hi)-ref)/stride,-.8,.8).tolist()
        if x is not None:
            sensed.append([tick,*x,*q,*ref,*u])
        controls.append(u)
        _,_,ended,capped,info=a.exchange(u,mpc.FAR_COMMAND)
        if a.rival_returns>=leg+1 and any(e["kind"]=="legal_return" and e["side"]==1 for e in info["events"]):
            outcome=True;break
        if ended:
            outcome=False;break
        if capped:break
    ps=a.point_placements()
    offer=ps["delivered"][leg] if len(ps["delivered"])>leg else None
    request=ps["requested"][leg] if len(ps["requested"])>leg else None
    if a.legal_returns<43:
        raise ValueError("The original 42nd own landing changed during the blind intervention")
    return dict(controls=controls,observations=sensed,new_offer=offer,requested=request,armed_tick=episode.armed_tick,
                first_inbound_tick=episode.inbound,encoder_at_inbound=episode.ready,timed_out=episode.timed_out,
                legal=bool(offer and offer["near_landed"]),censored=outcome is None or offer is None,
                short_score=a.legal_returns-1,events_sha256=digest_json(a.events),
                notes=actor.notes,new_used_memory=actor.priors.digest())


def run(plan,out,split):
    reg=_read(plan)
    d=_read(ARCHIVE)
    case,old=terminal(d)
    if reg["source_sha256"]!=_hash(ARCHIVE) or reg["teacher_sha256"]!=_hash(PROOF):
        raise ValueError("Changed exact record or teacher")
    m=_read(MODEL)
    bank=Local(m["rows"],[r["context"] for r in m["easy"]])
    home=np.asarray(d["ready"])
    record=dict(schema="tt49-blind-physical-comparison-1",plan_sha256=_hash(plan),split=split,
                original_ready=home.tolist(),leg=case["leg"],anchor=reg["anchor"],rows=[])
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    with mpc.centered(home):
        a=mpc.arena()
        res=mpc.resources(a)
        image,rig,*_=res
        with patch.object(mpc,"Fragments",lambda f,e:Carried(f,e,practice=[course.PRACTICE],held={case["id"]})):
            actor=Repair(mpc.BANK,raw,readout=bank,examples=(),coverage=fresh.shot.Coverage(_read(fresh.RISK)),
                         budget=d["budget"],variant=d["variant"])
        try:
            doc={**d,"rows":[None]*len(_read(season.CARRY)["rows"])+d["rows"]}
            course.resume(actor,BANK,doc)
            history=prefix(a,case["card"],old,reg["anchor"],image,a._arm_ids[1][2],window=32)
            saved=State(a)
            observed=Observer(a,rig,actor,history)
            if len(a.partner.requests)!=case["leg"]:
                raise ValueError("This physical checkpoint is not before the designated new draw")
            record["archived_prefix_event_sha256"]=digest_json(a.events)
            record["memory"]=actor.priors.digest()
            for c in reg["roster"]:
                if split!="both" and c["split"]!=split:continue
                condition_start=process_time()
                branches={}
                for mode in ("stock","blind"):
                    saved.restore(a)
                    a.card=dict(case["card"],reply_seed=c["stream"])
                    eye=observed.restore(actor)
                    actor.mode="stock"
                    result=race(a,actor,eye,res,old,case,reg,mode)
                    if not result["requested"] or result["requested"]["target"]!=c["goal"]:
                        raise ValueError("Did not make exactly the assigned broad next draw")
                    branches[mode]=result
                p,q=(branches[mode]["new_offer"] for mode in ("stock","blind"))
                eq=(p is None and q is None) or bool(p and q and
                     all(same_trace(p[k],q[k]) for k in ("far_contact","near_bounce")))
                past_req=same_trace(branches["stock"]["requested"],branches["blind"]["requested"])
                if not eq or not past_req:
                    raise ValueError("Preparation altered the physical outgoing or target draw")
                record["rows"].append(dict(c,branches=branches,same_physical_incoming=eq,
                      actual_projected_m=None if not p or not p["receive_line"] else
                                    projected_height(p["receive_line"]),cpu_s=process_time()-condition_start))
                _write(out,record)
                print(c["split"],c["stream"],[(mode,b["legal"],b["censored"],
                      b["armed_tick"],b["first_inbound_tick"]) for mode,b in branches.items()],flush=True)
        finally:
            actor.close();image.close();a.close()
    record["complete"]=True
    _write(out,record)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","run"))
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--output",type=Path,default=ROOT/"native.json.gz")
    p.add_argument("--split",choices=("training","held","both"),default="both")
    a=p.parse_args()
    a.plan.parent.mkdir(parents=True,exist_ok=True)
    freeze(a.plan) if a.mode=="plan" else run(a.plan,a.output,a.split)
