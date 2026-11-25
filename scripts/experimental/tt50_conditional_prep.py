"""Integrate causal opponent-video forecasts with direct arm preparation and observed strike readout."""

import argparse
from pathlib import Path
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash,_read,_write
from .tt19_view_full_game import digest_json
from .tt37_opponent import draw
from .tt37_bodycoach_windows import Reach
from .tt39_carry_joints import Carried
from .tt40_long_low import preparation,same_trace
from .tt44_paired_credit import State,prefix
from .tt48_exact_repair import Repair,MODEL as DIRECT_MODEL
from .tt44_continuous_credit_fit import Local
from .tt48_terminal_repair import ARCHIVE,BANK,terminal
from .tt49_terminal_blind import Observer,ROOT as BLIND,PROOF
from .tt50_conditional_training import ContinuousMap,ROOT,TRAIN,P_SCALE
from .tt50_far_readout import Predictor,ROOT as VISION_ROOT

CAMERA=VISION_ROOT/"model.npz"
PLAN=ROOT/"plan.json"
SKILL=ROOT/"optical-skill.json.gz"
MODES=("stock","map","body-only","body-learned")


def plan(path):
    d=_read(ARCHIVE); case,old=terminal(d)
    baseline=BLIND/"compromise-protocol.json"
    older=BLIND/"protocol.json"
    a,b=_read(baseline),_read(older)
    xs=[dict(split="exact-hard-source",stream=case["card"]["reply_seed"],
             goal=old["placements"]["requested"][-1]["target"])]
    xs+=[dict(r,split="disjoint-compromise") for r in a["roster"] if r["split"]=="held"]
    xs+=[dict(r,split="disjoint-ordinary") for r in b["roster"] if r["split"]=="held"][:2]
    _write(path,dict(schema="tt50-conditional-physical-protocol-1",anchor=b["anchor"],
                     keep_old_commands_through=b["previous_swing_controls_through_tick"],
                     source_sha256=_hash(ARCHIVE),teacher_sha256=_hash(PROOF),rows=xs,
                     role="Original exact hard-source diagnostic plus five unopened-to-this-motor-fit held reply streams."))


def deep_coordinates(f):
    x,z,vx,vy,vz=map(float,f)
    t=.43/max(1.35,-.98*vy)
    return x+t*vx,z+t*vz-4.905*t*t


def bounded_forecast(f):
    return bool(np.shape(f)==(5,) and np.isfinite(f).all() and abs(f[0])<.51
                and .79<f[1]<1.29 and abs(f[2])<2.7 and -6.<f[3]<-1.9 and abs(f[4])<2.35)


def hard_gate(gate,f,x):
    lateral,height=deep_coordinates(f)
    return bool(gate<=0. and -.40<x[1]<.24 and -.28<lateral<.025
                and .891<height<.987 and x[4]>.9 and -.25<f[4]<.82)


class Taught:
    def __init__(self,packet):
        self.rows=[]
        for sample in packet["rows"]:
            if sample["reward"]!="physical legal and answered" or sample["x_at_receive_line"] is None:
                continue
            log=sample["observations"]
            p=sample["native_touch"]
            cues=[r for r in log if r["x"] is not None and r["x"][4]<-.9
                  and r["tick"]>=sample["mode_start"] and r["x"][1]>p[1]-.06]
            if not cues: continue
            cue=min(cues,key=lambda r:abs(r["x"][1]-p[1]))
            touch=cue["tick"]
            relevant=[r for r in log if touch-23<=r["tick"]<=touch+10]
            self.rows.append(dict(name=sample["name"],p=np.array(sample["x_at_receive_line"]),
                                  relative=np.array([r["tick"]-touch for r in relevant]),
                                  refs=np.array([r["next_ref"] for r in relevant]),depth=p[1]))

    def proposals(self,forecast,path,x):
        options=[]
        for r in self.rows:
            d=np.linalg.norm((forecast-r["p"])/[.13,.073,.75,.84,.88])
            if d>2.0: continue
            eta=(r["depth"]-x[1])/min(x[4],-1.4)
            nearby=path[abs(path[:,2]-r["depth"])<.075]
            if len(nearby):
                eta=nearby[np.argmin(abs(nearby[:,2]-r["depth"])),0]
            if not .030<eta<.330: continue
            for shift in (-2.,0.,2.):
                phase=np.arange(mpc.HORIZON)+1-eta/mpc.DT+shift
                g=np.column_stack([np.interp(phase,r["relative"],a) for a in r["refs"].T])
                options.append((r["name"],g))
        return options


class Arm(Repair):
    def __init__(self,*args,predictor_path,affine,taught,checker,waypoint,approach,**kw):
        super().__init__(*args,**kw)
        self.predictor=Predictor(predictor_path)
        self.affine=affine
        self.taught=taught
        self.checker=checker
        self.waypoint=np.asarray(waypoint)
        self.approach=np.asarray(approach)
        self.policy_mode="stock"

    def begin(self):
        super().begin()
        self.predictor.reset()
        self.y_out=None
        self.forecasts=[]
        self.exposure=None
        self.nudge=None
        self.plan=None
        self.plan_t=None
        self.hard_active=False
        self.closed=False
        self.seen_inbound_at=None
        self.next_try=-999
        self.committed=False
        self.last_empty=False
        self.recovering=False
        self.shots=[]

    def cold_after_contact(self,frame,x,tick):
        # Still feed the same real exposures, but no preview can command
        # the continuation unless its causal outbound gate was actually seen.
        self.predictor.update(frame,x,tick)

    def available(self,ref,x,gate):
        n=round(np.clip((.93-x[1])/max(x[4],1.7),.26,.65)*60+28)
        n=int(np.clip(n,40,65))
        # A lower bound on installed reference travel, with a small settled margin.
        qmax=np.max(abs(np.clip(self.approach,self.checker.ik.low,self.checker.ik.high)-ref[:5]))
        if qmax/4.16+.075>n*mpc.DT:return None
        goal=preparation(ref[:5],self.waypoint,self.approach,n,ref[-1],brisk=True)
        qs=np.clip(goal[np.linspace(0,len(goal)-1,12,dtype=int),:5],
                   self.checker.ik.low,self.checker.ik.high)
        if not all(self.checker.clear(q) for q in (ref[:5],*qs)):return None
        return goal

    def prepare(self,tick,x,q,ref,g,forecast):
        if forecast is None or x is None or x[4]<=.9:return
        gate,f=forecast
        note=dict(tick=tick,gate=float(gate),x=f.tolist(),used="none")
        self.forecasts.append(note)
        if not bounded_forecast(f):
            return
        self.exposure=tick
        if self.policy_mode=="map":
            delta,support=self.affine.query(f,q,ref)
            if support>0 and self.checker.route(q[:5],np.clip(g[:5]+.58*delta,self.checker.ik.low,self.checker.ik.high),
                                               steps=6):
                self.nudge=np.r_[.58*support*delta,0.]
                note.update(used="map",support=support,wanted_delta=self.nudge.tolist())
            return
        if self.policy_mode not in ("body-only","body-learned") or self.closed:
            return
        # Once a reliable early apparent low has opened the fixed route,
        # a later clear ordinary prediction can still cancel it.
        lateral,high=deep_coordinates(f)
        if self.plan is not None and (lateral>.12 or high>1.048) and gate<=.45:
            self.plan=None
            self.hard_active=False
            self.closed=True
            self.recovering=True
            note["used"]="visible outgoing disagreement"
            return
        if self.plan is not None or not hard_gate(gate,f,x):return
        goal=self.available(ref,x,gate)
        if goal is None:
            note["used"]="not within bounded route"
            return
        self.plan=goal
        self.plan_t=tick
        self.hard_active=True
        note.update(used="direct-body",duration_ticks=len(goal))

    def visual_path(self):
        if not self.engine.sync():
            return np.empty((0,10))
        t=self.engine.twin
        return t.partner.flight.path(t.ball,t.velocity,t.spin,1,bounced=self.engine.bounced,horizon=.67)

    def try_taught(self,tick,x,path):
        line=path[abs(path[:,2]+1.10)<.040]
        if self.taught is None or not len(line):return None
        a=line[np.argmin(abs(line[:,2]+1.10))]
        projected=np.r_[a[1],a[3:7]]
        if not bounded_forecast(projected):return None
        _,height=deep_coordinates(projected)
        if not -.32<projected[0]<.14 or height>1.020:return None
        b=self.taught.proposals(projected,path,x)
        if not b:return None
        record=dict(tick=tick,approach=projected.tolist(),tested=len(b),nominal_legal=0)
        self.shots.append(record)
        options=[g for _,g in b]
        a,_,trace,_,_=self.trial(options,traces=True)
        for i in np.argsort(a)[:3]:
            change=self.transport(trace[i],face=True)
            if change is not None:
                options.append(options[i]+.7*change)
        loss,legal,_,grade,_=self.trial(options,long=True)
        keep=np.flatnonzero(legal & (grade[:,5]>.012))
        if not len(keep):return None
        keep=keep[np.argsort(loss[keep])[:4]]
        pick,support,cost=self.robust([options[k] for k in keep],keep,loss[keep])
        record.update(nominal_legal=int(legal.sum()),public_support=int(support))
        if support<1:return None
        idx=int(np.where(keep==pick)[0][0])
        at=keep[idx]
        self.best=np.array(options[at])
        self.tick=tick
        self.lease=min(len(self.best),max(4,round((grade[at,1]+.08)/mpc.DT)))
        self.committed=True
        record.update(chosen=int(at),contact_eta=float(grade[at,1]))
        return self.best[0]

    def normal_fallback(self,q,ref,plain):
        if not self.recovering:
            return plain
        lo,hi=self.checker.ik.low,self.checker.ik.high
        goal=np.clip(plain[:5],lo,hi)
        if self.checker.route(q[:5],goal,steps=6):
            if np.max(abs(q[:5]-goal))<.21:
                self.recovering=False
            return plain
        # Return along the last known side of the route, or hold commands.
        # Target clipping and the real plant's collision guard remain in force.
        step=ref[:5]+np.clip(self.waypoint-ref[:5],-.108,.108)
        step=np.clip(step,lo,hi)
        return np.r_[step,ref[-1]] if self.checker.route(q[:5],step,steps=5) else ref.copy()

    def blended(self,tick,frame,eye,x,q,ref,home):
        outcome=self.predictor.update(frame,x,tick)
        if x is not None and x[4]>.9:
            if self.seen_inbound_at is not None:
                self.plan=None
                self.nudge=None
                self.hard_active=False
                self.seen_inbound_at=None
                self.closed=False
                self.committed=False
            self.y_out=tick
        if (self.y_out is not None and x is not None and x[4]<-.9 and self.seen_inbound_at is None
                and abs(x[1])<1.72 and tick-self.y_out<40):
            self.seen_inbound_at=tick
            self.nudge=None
            if self.policy_mode=="body-only":
                self.closed=True
                self.recovering=self.plan is not None
                self.hard_active=False
        g=self.get_base(tick,x,q,ref,home)
        if self.committed and 0<=tick-self.tick<self.lease:
            self.engine.observe(tick,eye,q,ref)
            return self.best[tick-self.tick]
        if self.seen_inbound_at is not None:
            if (self.policy_mode=="body-learned" and self.hard_active and x is not None
                    and x[4]<-.9 and -.96<x[1]<-.39 and tick>=self.next_try):
                self.next_try=tick+3
                self.engine.observe(tick,eye,q,ref)
                path=self.visual_path()
                line=path[abs(path[:,2]+1.53)<.038]
                if len(line) and (line[0,3]>1.031 or abs(line[0,1])>.29):
                    self.hard_active=False
                    self.recovering=True
                stock_lease=(self.best is not None and 0<=tick-self.tick<self.lease and self.notes
                             and self.notes[-1].get("robust",0)>0)
                if stock_lease:
                    self.hard_active=False
                    self.recovering=False
                if self.hard_active and len(path):
                    chosen=self.try_taught(tick,x,path)
                    if chosen is not None:return chosen
                with patch.object(self.engine,"observe",lambda *a:None):
                    plain=super().step(tick,eye,x,q,ref,g)
            else:
                plain=super().step(tick,eye,x,q,ref,g)
            if self.hard_active and x is not None and x[1]<-1.06:
                self.hard_active=False
                self.recovering=True
            if self.hard_active and x is not None and x[1]>-1.06 and self.plan is not None:
                return self.plan[min(len(self.plan)-1,tick-self.plan_t)]
            return self.normal_fallback(q,ref,plain)
        plain=super().step(tick,eye,x,q,ref,g)
        self.prepare(tick,x,q,ref,plain,outcome)
        if self.hard_active and self.plan is not None:
            if tick-self.plan_t<=len(self.plan)+10:
                return self.plan[min(len(self.plan)-1,tick-self.plan_t)]
            self.hard_active=False
            self.recovering=True
        if self.nudge is not None and self.exposure is not None and tick-self.exposure<28:
            return plain+self.nudge
        return self.normal_fallback(q,ref,plain)


def run_offer(a,arm,eye,res,original,case,protocol):
    camera,rig,stride,lo,hi=res
    axes=a._arm_ids[1][2]
    begin=protocol["anchor"]
    ring=[];controls=[]
    outcome=None
    for tick in range(begin,begin+245):
        q=a.joints.copy();ref=a.data.ctrl[axes].copy()
        frame=camera.capture() if tick%2==0 else None
        x=mpc.observed(eye,tick,frame,q)
        if tick<=protocol["keep_old_commands_through"]:
            arm.cold_after_contact(frame,x,tick)
            arm.engine.observe(tick,eye.eye,q,ref)
            u=original["actions"][tick]
        else:
            desired=arm.blended(tick,frame,eye.eye,x,q,ref,arm.priors.home)
            u=np.clip((np.clip(desired,lo,hi)-ref)/stride,-.8,.8).tolist()
        controls.append(u)
        if x is not None:ring.append([tick,*x,*q,*ref,*u])
        _,_,done,capped,info=a.exchange(u,mpc.FAR_COMMAND)
        if a.rival_returns>=43 and any(e["kind"]=="legal_return" and e["side"]==1 for e in info["events"]):
            outcome=True;break
        if done:
            outcome=False;break
        if capped:break
    got=a.point_placements()
    offer=got["delivered"][case["leg"]] if len(got["delivered"])>case["leg"] else None
    requested=got["requested"][case["leg"]] if len(got["requested"])>case["leg"] else None
    return dict(legal=bool(offer and offer["near_landed"]),offer=offer,request=requested,censored=outcome is None,
                first_inbound_rgb=arm.seen_inbound_at,forecast=arm.forecasts,learned_strike_trials=arm.shots,
                motion_choices=arm.approaches,commands=controls,sensory=ring,notes=arm.notes,
                physical_event_digest=digest_json(a.events),actual_own=a.legal_returns-1,
                terminal_reason=next((e.get("reason") for e in a.events[::-1] if e["kind"]=="point"),None))


def match(a,b):
    x,y=a["offer"],b["offer"]
    return ((x is None and y is None) or bool(x and y and all(
         same_trace(x[k],y[k]) for k in ("far_contact","near_bounce"))))


def run(protocol,output,predict,skill,modes,limit):
    reg=_read(protocol);src=_read(ARCHIVE)
    case,old=terminal(src)
    if reg["source_sha256"]!=_hash(ARCHIVE) or reg["teacher_sha256"]!=_hash(PROOF):
        raise ValueError("Changed before-training physical trajectory or motor proof")
    results=dict(schema="tt50-conditional-prep-native-1",plan_sha256=_hash(protocol),vision_sha256=_hash(predict),
                 learned_arm_sha256=_hash(TRAIN),skill_sha256=None if skill is None else _hash(skill),
                 roster=reg["rows"][:limit],modes=list(modes),rows=[],actor_source_sha256=_hash(Path(__file__)))
    motor=ContinuousMap(_read(TRAIN))
    physical=next(r for r in _read(PROOF)["rows"] if r["physical"]["answered"] and r["exact_native_replay"])
    q=physical["port"]
    taught=Taught(_read(skill)) if skill is not None else None
    policy=_read(DIRECT_MODEL)
    local=Local(policy["rows"],[r["context"] for r in policy["easy"]])
    home=np.asarray(src["ready"])
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    with mpc.centered(home):
        a=mpc.arena()
        cad=mpc.arena(detail="cad")
        watcher=Reach(cad,home,home)
        res=mpc.resources(a)
        camera,rig,*_=res
        with patch.object(mpc,"Fragments",lambda f,e:Carried(f,e,practice=[course.PRACTICE],held={case["id"]})):
            arm=Arm(mpc.BANK,raw,readout=local,examples=(),coverage=fresh.shot.Coverage(_read(fresh.RISK)),
                    budget=src["budget"],variant=src["variant"],predictor_path=predict,affine=motor,taught=taught,
                    checker=watcher,waypoint=q["waypoint"],approach=q["approach"])
        try:
            past={**src,"rows":[None]*len(_read(season.CARRY)["rows"])+src["rows"]}
            course.resume(arm,BANK,past)
            history=prefix(a,case["card"],old,reg["anchor"],camera,a._arm_ids[1][2],window=32)
            saved=State(a)
            camera0=Observer(a,rig,arm,history)
            if len(a.partner.requests)!=case["leg"]:
                raise ValueError("Physical checkpoint is after its counterfactual draw")
            results["unchanged_outgoing_sha256"]=digest_json(a.events)
            for task in results["roster"]:
                at=process_time()
                variants={}
                for key in modes:
                    saved.restore(a)
                    a.card=dict(case["card"],reply_seed=task["stream"])
                    eye=camera0.restore(arm)
                    arm.policy_mode=key
                    variant=run_offer(a,arm,eye,res,old,case,reg)
                    if not variant["request"] or variant["request"]["target"]!=task["goal"]:
                        raise ValueError("Altered official TT37 broad requested reply")
                    variants[key]=variant
                ref=variants.get("stock",next(iter(variants.values())))
                row=dict(task,variants=variants,actual_offer_matches={k:match(ref,v) for k,v in variants.items()},
                         cpu_s=process_time()-at)
                results["rows"].append(row)
                _write(output,results)
                print(task["stream"],[(k,v["legal"],row["actual_offer_matches"][k],
                         [b["used"] for b in v["forecast"] if b["used"]!="none"],
                         int(len(v["learned_strike_trials"]))) for k,v in variants.items()],flush=True)
        finally:
            arm.close();camera.close();a.close();cad.close()
    results["complete"]=True
    _write(output,results)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","run"))
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--output",type=Path,default=ROOT/"physical.json.gz")
    p.add_argument("--predictor",type=Path,default=CAMERA)
    p.add_argument("--skill",type=Path,default=SKILL)
    p.add_argument("--limit",type=int,default=6)
    p.add_argument("--modes",choices=MODES,nargs="+",default=MODES)
    a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if a.mode=="plan":
        plan(a.plan)
    else:
        skill=a.skill if a.skill.is_file() else None
        if "body-learned" in a.modes and skill is None:
            raise ValueError("Body+observed motion needs physical optical skill bank")
        run(a.plan,a.output,a.predictor,skill,a.modes,a.limit)
