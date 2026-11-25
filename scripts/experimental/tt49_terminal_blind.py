"""Test the same visible-outbound preparation across fresh native reply draws."""

import argparse
from copy import deepcopy
from collections import deque
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
        self.start_reference=None
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
                self.start_reference=ref.copy()
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
        idx=min(tick-self.armed_tick,len(self.motion)-1)
        if self.mode=="short24":
            return self.motion[min(idx,23)]
        if self.mode=="partial":
            # Keep ordinary pan and lift recovery. Borrow only a fraction
            # of the taught elbow-and-wrist motion, independent of any lane.
            step=self.motion[idx]-self.start_reference
            return np.asarray(original)+step*np.array([0.,0.,.31,.12,.27,0.])
        return self.motion[idx]


def race(a,actor,eye,res,stock,case,reg,condition,extra=0):
    image,rig,stride,lo,hi=res
    control_channels=a._arm_ids[1][2]
    start=reg["anchor"]
    controls=[]; sensed=[]; episode=Blind(reg["joint_teaching"],condition,
                                         reg["previous_swing_controls_through_tick"]+1)
    home=np.asarray(actor.priors.home)
    leg=case["leg"];outcome=None
    first_score=False
    score_before=42
    ring=deque(maxlen=150)
    observed_follow_full=False
    for tick in range(start,start+240+140*extra):
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
        if extra:
            sent=np.clip(ref+np.asarray(u)*stride,lo,hi)
            ring.append((tick,x,q,ref,sent))
        _,_,ended,capped,info=a.exchange(u,mpc.FAR_COMMAND)
        if a.rival_returns>=leg+1 and any(e["kind"]=="legal_return" and e["side"]==1 for e in info["events"]):
            first_score=True
            outcome=True
            scored=a.legal_returns-1
            if extra and scored>score_before:
                actor.priors.keep(list(ring),("held-blind-reply",51,scored))
                score_before=scored
            if scored>=43+extra:
                observed_follow_full=bool(extra)
                break
        if ended:
            outcome=first_score
            observed_follow_full=bool(extra and first_score)
            break
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
                notes=actor.notes,new_used_memory=actor.priors.digest(),
                follow_assigned=extra,follow_score=max(0,a.legal_returns-44),follow_observed_complete=observed_follow_full,
                terminal_reason=next((e.get("reason") for e in reversed(a.events) if e["kind"]=="point"),None))


def run(plan,out,split,reuse_stock=None,chosen=None,extra=0):
    reg=_read(plan)
    d=_read(ARCHIVE)
    case,old=terminal(d)
    if reg["source_sha256"]!=_hash(ARCHIVE) or reg["teacher_sha256"]!=_hash(PROOF):
        raise ValueError("Changed exact record or teacher")
    modes=reg.get("variants",["stock","blind"])
    if chosen is not None:
        if split!="held" or chosen not in modes[1:]:
            raise ValueError("A frozen trained variant may only narrow a held schedule")
        modes=["stock",chosen]
    stock_cache={} if reuse_stock is None else {r["stream"]:r for r in _read(reuse_stock)["rows"]
                                               if r["split"]=="training"}
    m=_read(MODEL)
    bank=Local(m["rows"],[r["context"] for r in m["easy"]])
    home=np.asarray(d["ready"])
    record=dict(schema="tt49-blind-physical-comparison-1",plan_sha256=_hash(plan),split=split,
                original_ready=home.tolist(),leg=case["leg"],anchor=reg["anchor"],variants=modes,rows=[],continuation_extra=extra,
                reused_stock_sha256=None if reuse_stock is None else _hash(reuse_stock))
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
            start_bank=list(actor.priors.online)
            for c in reg["roster"]:
                if split!="both" and c["split"]!=split:continue
                condition_start=process_time()
                branches={}
                for mode in modes:
                    old_control=stock_cache.get(c["stream"])
                    if mode=="stock" and old_control is not None:
                        if old_control["goal"]!=c["goal"]:
                            raise ValueError("Cannot reuse a different native draw")
                        branches[mode]=old_control["branches"]["stock"]
                        continue
                    saved.restore(a)
                    if extra:
                        actor.priors.online.clear()
                        actor.priors.online.extend(start_bank)
                    a.card=dict(case["card"],reply_seed=c["stream"])
                    eye=observed.restore(actor)
                    actor.mode="stock"
                    result=race(a,actor,eye,res,old,case,reg,mode,extra)
                    if not result["requested"] or result["requested"]["target"]!=c["goal"]:
                        raise ValueError("Did not make exactly the assigned broad next draw")
                    branches[mode]=result
                stock_branch=branches["stock"]
                p=stock_branch["new_offer"]
                eq=True
                for mode in modes[1:]:
                    change=branches[mode]
                    q=change["new_offer"]
                    exact=(p is None and q is None) or bool(p and q and
                          all(same_trace(p[k],q[k]) for k in ("far_contact","near_bounce")))
                    eq &= exact and same_trace(stock_branch["requested"],change["requested"])
                if not eq:
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
    p.add_argument("--reuse-stock",type=Path)
    p.add_argument("--chosen",choices=("short24","partial"))
    p.add_argument("--continuation",type=int,choices=(0,1,2),default=0)
    a=p.parse_args()
    a.plan.parent.mkdir(parents=True,exist_ok=True)
    freeze(a.plan) if a.mode=="plan" else run(a.plan,a.output,a.split,a.reuse_stock,a.chosen,a.continuation)
