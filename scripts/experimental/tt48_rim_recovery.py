"""Transfer physically earned corrections to optically predicted low glancing contacts."""
import argparse
from copy import copy
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import DRAG, LIFT, NOMINAL_NO_LOAD
from .tt37_opponent import draw
from .tt39_carry_joints import Carried
from .tt48_contact_patch import ROOT, SOURCE, ease

PRACTICE = ROOT / "causal.json.gz"
PLAN = ROOT / "held.protocol.json"
STOCK = season.ROOT / "before.json.gz"
DIRECT = mpc.MPC


def advice(receipt, archive):
    if not receipt["complete"] or _hash(SOURCE) != receipt["source_sha256"]:
        raise ValueError("Incomplete reference practice")
    rows = []
    for r in receipt["rows"]:
        arm = r["arms"].get("edge")
        if arm is None or not arm["landed"] or not arm["answered"] or arm["clearance"]["penetration_samples"]:
            continue
        original = r["arms"]["stock"]
        if original["landed"] or arm["actual"]["radius"] < .063:
            raise ValueError("Not an earned rim correction")
        stride = np.asarray(archive["rows"][r["index"]]["motor_audit"]["target_rate_limit_rad_s"])*mpc.DT/.8
        n = max(1, int(round(original["actual"]["when"]/mpc.DT)))
        contrast = np.cumsum((np.asarray(arm["controls"][:n])-np.asarray(original["controls"][:n]))*stride,axis=0)
        delta = contrast[-1]
        flank = float(np.sign(arm["actual"]["location"][0]))
        stock = next(e for e in original["events"] if e.get("collider")=="learner_racket")
        rows.append(dict(release=r["id"], flank=flank, delta=delta.tolist(),
                         measured_departure=stock["velocity"], source_arm_score=r["arms"]["edge"]["score"]["learner"]))
    return rows


def glance(f):
    return (f[1] > .5 and .064 < f[8] < .079 and -1.69 < f[6] < -1.40
            and .88 < f[7] < 1.085 and f[3] < 2.80)


def remap(delta, flank, target):
    out = np.asarray(delta).copy()
    if flank*target < 0:
        out[[0,4]] *= -1.
    return out


def alignment(engine, goals):
    """Inspect the collision constraint in the camera-initialized private plant."""
    predictor, twin = engine.predict, engine.twin
    if twin.launches or twin.card is not None:
        raise ValueError("Contact inspection requires the unlaunched optical plant")
    goals = np.asarray(goals)
    rates,_ = predictor.trajectories(goals,mpc.AIM,mpc.HORIZON*mpc.DT)
    model=copy(predictor.model)
    d=mujoco.MjData(model)
    force=predictor.model.actuator_forcerange[predictor.near_axes].copy()
    low,high,inc=twin._joint_low,twin._joint_high,twin._joint_increments
    p,v=twin._ball_joint,twin._ball_dof
    spin=np.zeros(3)
    blade, ball=twin._racket_ids[1],twin._geom_ids["ball"]
    result=np.full(len(goals),np.nan)
    for i,batch in enumerate(rates):
        mujoco.mj_copyData(d,model,twin.data)
        for action in batch:
            other=np.clip((predictor.other_home-d.ctrl[predictor.other])/predictor.micro,
                          -.8*inc[predictor.other],+.8*inc[predictor.other])
            for _ in range(predictor.micro):
                d.ctrl[predictor.axes]=np.clip(d.ctrl[predictor.axes]+inc[predictor.axes]*action,
                                               low[predictor.axes],high[predictor.axes])
                d.ctrl[predictor.other]=np.clip(d.ctrl[predictor.other]+other,
                                               low[predictor.other],high[predictor.other])
                vel=d.qvel[v:v+3]
                mujoco.mju_rotVecQuat(spin,d.qvel[v+3:v+6],d.qpos[p+3:p+7])
                d.xfrc_applied[twin._ball_body,:3]=model.body_mass[twin._ball_body]*(
                    -DRAG*np.linalg.norm(vel)*vel+LIFT*np.cross(spin,vel))
                d.xfrc_applied[twin._ball_body,3:]=-.18*model.body_inertia[twin._ball_body]*spin
                if twin.motor_envelope:
                    r=d.qvel[predictor.near_dofs]/NOMINAL_NO_LOAD
                    model.actuator_forcerange[predictor.near_axes]=force*np.clip(np.c_[1.+r,1.-r],0.,1.)
                mujoco.mj_step(model,d)
                contact=next((h for h in d.contact if h.efc_address>=0 and set(h.geom)=={blade,ball}),None)
                if contact is not None:
                    normal=d.geom_xmat[blade].reshape(3,3)[:,2]
                    result[i]=abs(float(normal@contact.frame[:3]))
                    break
            if np.isfinite(result[i]):
                break
    return result


class Rim(season.Player):
    def __init__(self,*args,corrections,**kw):
        super().__init__(*args,**kw)
        self.corrections=corrections

    def begin(self):
        super().begin()
        self.capture=None
        self.edge_actions=[]
        self.last_edge=-1000

    def trial(self, goals, *, long=False, traces=False):
        r=super().trial(goals,long=long,traces=traces)
        if self.capture is not None and not long:
            for g,f in zip(goals,r[4],strict=True):
                if glance(f):
                    self.capture.append((np.asarray(g).copy(),f.copy()))
        return r

    def search(self,tick,x,q,ref,base):
        start=len(self.notes)
        watch=bool(-1.31 < x[1] < -.55 and x[4] < -.9 and tick-self.last_edge >= 8)
        self.capture=[] if watch else None
        try:
            super().search(tick,x,q,ref,base)
            pool=self.capture
        finally:
            self.capture=None
        if not pool or len(self.notes)==start or self.notes[-1].get("legal_trials",0):
            return
        foci=[]
        def grade(row):
            g,f=row
            return abs(f[7]-1.003)*12.+abs(f[6]+1.55)*1.4+max(0.,1.1-f[3])*.12
        for path,fb in sorted(pool,key=grade):
            if any(np.max(abs(path-previous))<.035 for previous,_ in foci):
                continue
            foci.append((path, fb))
            if len(foci)==4:
                break
        self.last_edge=tick
        a0=alignment(self.engine,[g for g,_ in foci])
        foci=[(g,fb,dot) for (g,fb),dot in zip(foci,a0,strict=True)
              if np.isfinite(dot) and (dot<.86 or fb[3]<.65 or fb[4]<.8)]
        if not foci:
            return
        trials=[]
        tags=[]
        times=(1+np.arange(mpc.HORIZON))*mpc.DT
        for seed,fb,dot in foci:
            eta=float(np.clip((x[1]-fb[6])/-x[4],.07,.29))
            for note in self.corrections:
                shift=remap(note["delta"],note["flank"],fb[5])
                # The learned low wrist correction is a bounded reference
                # bias, not the archived motor tape or the native-contact time.
                for gain in (.68,1.03):
                    trial=seed+ease(times/max(.040,.84*eta))[:,None]*(gain*shift)
                    trials.append(trial)
                    tags.append((note["release"],gain,float(dot)))
        if not trials:
            return
        self.diag["rim_attempts"]+=1
        _,legal,_,y,feedback=DIRECT.trial(self,trials,long=True)
        use=np.flatnonzero(legal & (feedback[:,1]>.5))
        note=dict(tick=tick, generated=len(trials),nominal_legal=int(len(use)),issued=False)
        self.edge_actions.append(note)
        if not len(use):
            return
        loss,_=mpc.cost(y,feedback)
        pick=use[np.argsort(loss[use])[:4]]
        agree,risk=self.uncertainty([trials[i] for i in pick],loss[pick])
        a1=alignment(self.engine,[trials[i] for i in pick])
        passed=np.flatnonzero((agree>=1)&(a1>.74)&(a1+ .04 >= np.array([tags[i][2] for i in pick])))
        note["robust_candidates"]=int(len(passed))
        if not len(passed):
            return
        selected=int(passed[np.argmin(risk[passed]+.17*(2.-agree[passed]))])
        j=int(pick[selected])
        seconds=float(y[j,1])
        self.best=trials[j]
        self.tick=tick
        self.lease=min(mpc.HORIZON,max(mpc.LEASE,int(np.ceil(seconds/mpc.DT))+3))
        self.next=tick+self.lease
        self.diag["rim_issued"]+=1
        note.update(issued=True, donor=tags[j][0], gain=tags[j][1], agreement=int(agree[selected]),
                    modeled_face_alignment=[tags[j][2],float(a1[selected])],
                    eta_s=seconds, nominal_radius_m=float(feedback[j,8]), target_errors=y[j,3:10].tolist(),
                    quality=float(risk[selected]),lease=self.lease)
        self.notes[-1].update(chosen="earned_rim_motion",robust=note["agreement"],score=note["quality"],
                              contact_delay_s=seconds,target_errors=note["target_errors"])


def plan(out,n):
    base=_read(STOCK)
    assess=_read(season.PLAN)["assessment"]
    if base["cards"]!=assess or base["mode"]!="frozen" or base["cap"]!=240:
        raise ValueError("Season stock physical row order differs")
    cards=assess[:n]
    correction=advice(_read(PRACTICE),_read(SOURCE))
    if set(r["release"] for r in correction)&{c["id"] for c in cards}:
        raise ValueError("Physical donor leaks into assessment")
    _write(out,dict(cards=cards, cap=base["cap"],archived_control_sha256=_hash(STOCK),
                    source_sha256=_hash(Path(__file__)),practice_sha256=_hash(PRACTICE),
                    carry_sha256=_hash(season.CARRY), season_sha256=_hash(season.PLAN),
                    corrections=correction, rule="Unfiltered prefix of distinct fixed season assessment releases, identical measured opening and continuous reactive broad random opponent draws. At most 240 alternating physical returns as in stock. Experimental correction is frozen."))


def run(path,out):
    spec,baseline=_read(path),_read(STOCK)
    cards=spec["cards"]
    if (spec["archived_control_sha256"]!=_hash(STOCK) or spec["source_sha256"]!=_hash(Path(__file__))
            or spec["carry_sha256"]!=_hash(season.CARRY) or spec["practice_sha256"]!=_hash(PRACTICE)
            or spec["season_sha256"]!=_hash(season.PLAN) or cards!=baseline["cards"][:len(cards)]
            or not cards or spec["cap"]!=240
            or baseline["source_sha256"]!=_hash(mpc.BANK) or baseline["optical_sha256"]!=_hash(course.OPTIC)
            or baseline["risk"]!=_hash(fresh.RISK)):
        raise ValueError("Changed plan, archive, or causal preparation context")
    receipt=dict(cards=cards,cap=spec["cap"],ready=baseline["ready"],rows=[],complete=False,
                 checkpoint_before=_hash(season.CARRY.with_suffix("").with_suffix(".npz")),
                 protocol_sha256=_hash(path),control_sha256=_hash(STOCK))
    out.parent.mkdir(exist_ok=True,parents=True)
    if out.exists():
        last=_read(out)
        if any(last[k]!=v for k,v in receipt.items() if k not in ("rows","complete")):
            raise ValueError("Checkpoint protocol differs")
        receipt=last
    ids={c["id"] for c in cards}
    with np.load(mpc.BANK) as f:
        raw=mpc.DeepField(f,"local")
    ready=baseline["ready"]
    coverage=fresh.shot.Coverage(_read(fresh.RISK))
    state=_read(season.CARRY)
    tape=season.CARRY.with_suffix("").with_suffix(".npz")
    with mpc.centered(ready):
        arena=mpc.arena()
        camera,rig,stride,lo,hi=mpc.resources(arena)
        with patch.object(mpc,"Fragments",lambda f,e: Carried(f,e,practice=[course.PRACTICE],held=ids)):
            model=Rim(mpc.BANK,raw,variant="elastic",budget=baseline["budget"],
                      coverage=coverage,corrections=spec["corrections"])
        try:
            course.resume(model,tape,state)
            for ix in range(len(receipt["rows"]),len(cards)):
                c,old=cards[ix],baseline["rows"][ix]
                if model.priors.digest()!=old["memory"]["before"]:
                    raise ValueError("Changed initial memory")
                row=mpc.play(arena,camera,rig,stride,lo,hi,ready,c,old["shot"],raw,model,
                             course.OPTIC,"frozen",spec["cap"],10)
                for leg, target in enumerate(row["placements"]["requested"]):
                    if target["target"]!=list(draw(c,leg,"random",None)[0]):
                        raise ValueError("Live partner target changed")
                d=next((j for j,(a,b) in enumerate(zip(old["actions"],row["actions"])) if a!=b),
                       None if len(old["actions"])==len(row["actions"]) else min(len(old["actions"]),len(row["actions"])))
                if d is None:
                    if digest_json(row["events"])!=digest_json(old["events"]):
                        raise ValueError("Equal movements produced different events")
                else:
                    issued=[x for x in model.edge_actions if x["issued"]]
                    if not issued or d < issued[0]["tick"]:
                        raise ValueError("A control changed before optical correction")
                    if digest_json([e for e in old["events"] if e["step"]<d])!=digest_json([e for e in row["events"] if e["step"]<d]):
                        raise ValueError("Physical events changed before any differing action")
                row.update(edge_actions=model.edge_actions,stock_learner=old["score"]["learner"],
                           first_action_diff=d,stock_event_sha256=digest_json(old["events"]))
                receipt["rows"].append(row)
                receipt["complete"]=len(receipt["rows"])==len(cards)
                _write(out,receipt)
                print(c["id"],old["score"]["learner"],row["score"]["learner"],
                      row["planner"].get("rim_attempts",0),row["planner"].get("rim_issued",0),d,flush=True)
        finally:
            model.close()
            camera.close()
            arena.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("task",choices=("plan","run"))
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--output",type=Path,default=ROOT/"held.json.gz")
    p.add_argument("--n",type=int,default=24)
    a=p.parse_args()
    if a.task=="plan":
        plan(a.plan,a.n)
    else:
        run(a.plan,a.output)
