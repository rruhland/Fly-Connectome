"""Invert measured first-contact velocity differences without an online contact simulator."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt36_impact import contacts
from .tt36_impact_refine import Frames
from .tt37_joint_capture import observed
from .tt37_joint_deep import DeepField
from .tt37_opponent import arena
from .tt37_visual_inertia import LearnedOptics
from .tt40_real_contact import CAMERA, DT, OUT, STOCK, gate

ROOT=Path("artifacts")
BANK=ROOT/"tt37/long-rgb.npz"
OPTICS=ROOT/"tt37/rebound.npz"
PLAN=ROOT/"tt39/hindsight-plan.json"
SCALE=np.array([.105,.14,.105,.85,1.12,1.15,.58,.72,.62,.57,.63,.25,.25,.25,.3,.27])
VEL=np.array([1.65,2.2,2.4])


def basis(ticks):
    elapsed=np.maximum(np.asarray(ticks,dtype=float),0.)
    rate=np.clip(elapsed/20.,0.,1.)
    ease=np.clip(elapsed/5.,0.,1.)
    return np.stack((ease*(1.-rate),ease*rate),axis=-1)


def project(raw,new):
    """Regress the actual issued difference on two smooth controls per joint."""
    if raw["impact"] is None or new["impact"] is None:
        return None
    t=min(raw["impact"]["time"],new["impact"]["time"])
    exact=min(len(raw["reference"]),len(new["reference"]),int(np.floor(t/DT-raw["at"])))
    if exact<8:
        return None
    ramp=basis(np.arange(1,exact+1))
    x=np.asarray(new["reference"])[:exact,:5]-np.asarray(raw["reference"])[:exact,:5]
    theta=np.linalg.solve(ramp.T@ramp+.09*np.eye(2),ramp.T@x)
    mismatch=float(np.sqrt(np.mean((x-ramp@theta)**2)))
    return theta.flatten(),mismatch,exact


def prepare(camera,output):
    source=_read(camera)
    records=[]
    for r in source["rows"]:
        stock=r["trials"]["raw"]
        first=stock["impact"]
        if first is None:
            continue
        y0=np.asarray(first["outgoing"])
        for key,branch in r["trials"].items():
            if key=="raw" or branch["impact"] is None:
                continue
            lo=project(stock,branch)
            if lo is None:
                continue
            theta,reconstruction,support=lo
            delta=np.asarray(branch["impact"]["outgoing"])-y0
            if np.linalg.norm(theta)<.035 or np.linalg.norm(delta)<.10:
                continue
            records.append(dict(release=r["card"]["id"],mode=key,state=r["context"],theta=theta.tolist(),
                                base=y0.tolist(),delta=delta.tolist(),score=branch["score"],
                                projection_rmse_rad=reconstruction,ticks=support))
    if not records:
        raise ValueError("Physical stereo pairs did not yield any observed command consequences")
    doc=dict(camera_sha256=_hash(camera),records=records,groups=sorted({r["release"] for r in records}),
             note="Contrast fit from exact real action prefixes. Both label velocities came from RGB. Native grading does not enter fit.")
    _write(output,doc)
    print("usable actual contact contrasts",len(records),"release groups",len(doc["groups"]),flush=True)


class Effect:
    """Local ridge derivative on action contrasts and empirical camera velocity goals."""

    def __init__(self,archive,witness=()):
        rows=archive["records"]
        self.x=np.array([r["state"] for r in rows])/SCALE
        self.base=np.array([r["base"] for r in rows])
        self.com=np.array([r["theta"] for r in rows])/.23
        self.response=np.array([r["delta"] for r in rows])/VEL
        self.good=np.array([r["score"]>0 for r in rows])
        _, group, counts=np.unique([r["release"] for r in rows],return_inverse=True,return_counts=True)
        self.prior=1./counts[group]
        self.quality=np.array([1./(1.+(r["projection_rmse_rad"]/.12)**2) for r in rows])
        self.witness=list(witness)
        self.secants=archive.get("owned_secants",[])

    def correction(self,x,mode):
        d=np.linalg.norm(self.x-np.asarray(x)/SCALE,axis=1)
        nearest=float(d.min())
        if mode=="fixed" or nearest>2.9 or not np.any(self.good):
            return np.zeros((2,5)),dict(nearest=nearest)
        w=np.exp(-.5*(d/1.05)**2)*self.prior*self.quality
        present=w@self.base/w.sum()
        observed_miss=None
        if mode=="carry":
            if not self.witness:
                return np.zeros((2,5)),dict(nearest=nearest,miss_support=False)
            nearest_own,i=min((np.linalg.norm((np.asarray(o["context"])-x)/SCALE),i)
                              for i,o in enumerate(self.witness))
            if nearest_own>.58:
                return np.zeros((2,5)),dict(nearest=nearest,miss_support=False)
            observed_miss=float(nearest_own)
            present=np.asarray(self.witness[i]["outgoing"])
        legal=w*self.good
        target=legal@(self.base+self.response*VEL)/legal.sum()
        feature=self.com
        model=np.linalg.solve(feature.T@(w[:,None]*feature)+.075*np.eye(10),feature.T@(w[:,None]*self.response)).T
        secant_weight=0.
        for case in self.secants:
            ds=np.linalg.norm((x-np.asarray(case["context"]))/SCALE)
            trust=float(np.clip((.65-ds)/.36,0.,1.))
            if trust:
                step=np.asarray(case["executed_contrast"])/.23
                error=np.asarray(case["measured_change"])/VEL-model@step
                model+=trust*np.outer(error,step)/(step@step+.055)
                present=(1.-trust)*present+trust*np.asarray(case["observed_baseline"])
                secant_weight=max(secant_weight,trust)
        if mode=="empirical":
            aim=legal@feature/legal.sum()
        else:
            e=(target-present)/VEL
            aim=model.T@np.linalg.solve(model@model.T+.090*np.eye(3),e)
        strength=float(np.clip((2.9-nearest)/1.25,0.,1.))
        if mode=="reversed":
            aim=-aim
        cmd=np.clip(.23*strength*aim,-.26,.26).reshape(2,5)
        return cmd,dict(nearest=nearest,measured_miss_distance=observed_miss,
                        own_failure_secant_weight=secant_weight,
                        observed_base_mix_mps=present.tolist(),desired_observed_mps=target.tolist(),
                        predicted_change_mps=(VEL*(model@(cmd.ravel()/.23))).tolist())


def play(a,renderer,rig,stride,low,high,ready,card,shot,raw,model,mode,calibration,*,continue_point=False):
    a.launch(shot,card=card)
    a.partner.reset(seed=card["seed"])
    eye=Vision(rig,a.control_dt,raw=True)
    eye.eye=LearnedOptics(eye.eye,OPTICS)
    watch=Frames(rig)
    home=np.asarray(ready)
    ref=home.copy()
    trace=[]
    t0=None
    inbound=False
    active=True
    elapsed=0
    q0=None
    started=process_time()
    for i in range(a.max_steps if continue_point else min(480,a.max_steps)):
        q=a.joints
        frame=renderer.capture() if i%2==0 or t0 is not None and i<t0+42 else None
        opt=observed(eye,i,frame if i%2==0 else None,q)
        if frame is not None and (t0 is not None and i<t0+42 or
                                  t0 is None and opt is not None and opt[4]<-.9 and opt[1]<-.15):
            watch.observe(frame,i*DT,q)
        if opt is not None and opt[4]<-.9 and opt[1]<-.9:
            inbound=True
        desired=home
        if opt is not None and (opt[4]<-.9 or (inbound and opt[1]<.50)):
            goal,conf=raw.get(opt,q,reference=ref)
            if goal is not None:
                desired=home+conf*(goal-home)
        if t0 is None and opt is not None and opt[4]<-.9 and -1.05<opt[1]<-.545:
            t0=i
            q0=np.r_[opt,q[:5],ref[:5]-q[:5]]
            choice,details=model.correction(q0,mode)
        if t0 is not None:
            if opt is not None and opt[4]>.9:
                active=False
            if active and 0<=i-t0<28 and np.linalg.norm(choice)>1.e-7:
                shift=basis(i-t0+1)@choice
                desired=desired.copy()
                desired[:5]+=shift
                elapsed+=1
        u=np.clip((np.clip(desired,low,high)-ref)/stride,-.8,.8)
        ref=np.clip(ref+u*stride,low,high)
        _,_,done,capped,_=a.exchange(u,FAR_COMMAND)
        trace.append(u.tolist())
        if done or capped or not continue_point and a.legal_returns>1 or a.legal_returns+a.rival_returns-1>=200:
            break
    impacts=watch.estimates(calibration)
    first=next((p for p in impacts if t0 is not None and .025<p["time"]-t0*DT<.43),None)
    out=dict(shot=shot,opponent_seed=card["seed"],actions=trace,events=a.events,
             score=audit(a.events,opening_side=1),placements=a.point_placements(),
             motor_audit=a.motor_audit(),runtime_cpu_s=process_time()-started,
             first_camera_contact=first,modified_ticks=elapsed,
             camera=[[t,p.tolist(),None if b is None else b.tolist()] for t,p,b in watch.stereo],
             first_causal_tick=t0,context=None if q0 is None else q0.tolist())
    if t0 is not None:
        out.update(chosen_offset_rad=choice.tolist(),readout=details)
    out["native_evaluation"]=contacts(out)[:1]
    return out


def physical(checkpoint,deck,mode,out,limit,*,continue_point=False,witness_path=None):
    h=_read(checkpoint)
    witness=[]
    previous=_read(STOCK)
    by_card={c["id"]:r for c,r in zip(previous["cards"],previous["rows"],strict=True)}
    if deck=="practice":
        cards=[c for c in previous["cards"] if c["id"] in h["groups"]]
    else:
        cards=_read(PLAN)["fresh"]
        if {c["id"] for c in cards}&set(h["groups"]):
            raise ValueError("Held launch overlaps the contact effect calibration")
    if mode=="carry":
        if witness_path is None:
            raise ValueError("Carry requires a separate physically observed first-attempt exposure")
        prepared=_read(witness_path)
        if (prepared["mode"]!="fixed" or prepared["deck"]!=deck or prepared["cards"]!=cards
                or len(prepared["rows"])!=len(cards) or prepared["checkpoint_sha256"]!=_hash(checkpoint)):
            raise ValueError("Prior physical observations are not the complete frozen first-attempt roster")
        for r in prepared["rows"]:
            if not r["score"]["learner"] and r["context"] is not None and r["first_camera_contact"] is not None:
                witness.append(dict(context=r["context"],outgoing=r["first_camera_contact"]["outgoing"]))
        cards=[dict(c,reply_seed=4010765+133*i) for i,c in enumerate(cards)]
    model=Effect(h,witness)
    if limit:
        cards=cards[:limit]
    if continue_point and deck=="practice":
        cards=[dict(c,reply_seed=4010491 + 117*i) for i,c in enumerate(cards)]
    ready=previous["ready"]
    launches=measured_launches(Path("/tmp/tt25-rallies.json"))
    out.parent.mkdir(parents=True,exist_ok=True)
    document=dict(cards=cards,rows=[],ready=ready,mode=mode,deck=deck,checkpoint_sha256=_hash(checkpoint),
                  standard_opponent=True,stop_scope="terminal or 200 alternating" if continue_point else
                  "first scored own return or terminal",source_sha256=_hash(BANK),
                  optics_sha256=_hash(OPTICS))
    if witness_path is not None:
        document.update(witness_sha256=_hash(witness_path),observed_misses=len(witness),
                        carry_boundary="Repeat measured first deliveries with redrawn continuation seeds")
    with np.load(BANK) as b:
        raw=DeepField(b,"local")
    calibration=_read(CAMERA)["models"]["60"]
    with centered(ready):
        a=arena(protocol="random")
        rend,rig,stride,lo,hi=resources(a)
        try:
            for c in cards:
                r=play(a,rend,rig,stride,lo,hi,ready,c,launches[c["id"]],raw,model,mode,calibration,
                       continue_point=continue_point)
                if deck=="practice":
                    before=by_card[c["id"]]
                    t=gate(before)[0]
                    if r["actions"][:t]!=before["actions"][:t]:
                        raise ValueError("Same approach did not produce the same gates or commands")
                    if not r["modified_ticks"] and digest_json(r["events"])!=digest_json(before["events"]):
                        raise ValueError("Untouched practice did not exactly reproduce")
                document["rows"].append(r)
                _write(out,document)
                first=r["first_camera_contact"]
                print(mode,deck,c["id"],r["score"]["learner"],r["modified_ticks"],None if first is None else first["outgoing"],flush=True)
        finally:
            rend.close()
            a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    phases=p.add_subparsers(dest="phase",required=True)
    fit=phases.add_parser("fit")
    fit.add_argument("camera",type=Path)
    fit.add_argument("output",type=Path)
    test=phases.add_parser("physical")
    test.add_argument("checkpoint",type=Path)
    test.add_argument("deck",choices=("practice","fresh"))
    test.add_argument("mode",choices=("fixed","inverse","empirical","reversed","carry"))
    test.add_argument("output",type=Path)
    test.add_argument("--limit",type=int)
    test.add_argument("--continue-point",action="store_true")
    test.add_argument("--witness",type=Path)
    args=p.parse_args()
    if args.phase=="fit":
        prepare(args.camera,args.output)
    else:
        physical(args.checkpoint,args.deck,args.mode,args.output,args.limit,continue_point=args.continue_point,
                 witness_path=args.witness)
