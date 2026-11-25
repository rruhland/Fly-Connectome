"""Try to extract actionable angular evidence from ordinary RGB across a bounce."""

import argparse
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from . import tt38_joint_mpc as mpc
from . import tt42_camera_precision as dense
from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import BallFlight, _aero
from .tt29_vision import BOUNCE_Z
from .tt48_contact_patch import ROOT
from .tt48_contact_optical_check import CHECKS
from .tt48_optical_alternatives import outcome
from .tt48_precise_crop import CroppedStereo
from .tt48_precise_receive import Arc, DT, MICRO, aperture, camera_pixels, sphere_correct

WINDOW = ROOT / "bounce-window.json.gz"
S0=np.r_[[.0007,.0012,.0006],[.012,.018,.014],[34.,22.,28.]]
SIGHT=np.array([.00045,.00072,.00042])


def location(cal,pv,pixels):
    if not np.isfinite(pixels).all():
        return None
    model=np.asarray(pv[:3])[None,:]
    corrected=sphere_correct(cal,pixels[None,:],model)
    B,y=dense.matrix(cal,corrected,model)
    B=B.reshape(-1,3); y=y.ravel()
    point=np.linalg.solve(B.T@B,B.T@y)
    e=dense.image_residual(cal,point[None,:],corrected)
    if np.linalg.norm(point-model[0])>.098 or (e.max()*384/cal["dims"][0])>.52:
        return None
    return point


class BounceMap:
    """Independent public calibrated table, initialized only from image-derived p/v/spin."""

    def __init__(self,private):
        self.flight=BallFlight(private)

    def generate(self, s, ticks):
        model,d=self.flight.model,self.flight.data
        mujoco.mj_resetData(model,d)
        d.qpos[:]=np.r_[s[:3],1.,0.,0.,0.]
        d.qvel[:]=s[3:]
        w=np.zeros(3)
        mujoco.mj_forward(model,d)
        incoming=[]
        seen=False
        for _ in range(ticks):
            for i in range(MICRO):
                mujoco.mju_rotVecQuat(w,d.qvel[3:],d.qpos[3:])
                d.xfrc_applied[1]=_aero(d.qvel[:3],w,self.flight.drag,self.flight.lift)
                mujoco.mj_step(model,d)
                seen|=any(c.efc_address>=0 and (c.geom[0] in self.flight.surface
                          or c.geom[1] in self.flight.surface) for c in d.contact)
            mujoco.mju_rotVecQuat(w,d.qvel[3:],d.qpos[3:])
            incoming.append(np.r_[d.qpos[:3],d.qvel[:3],w,float(seen)])
        return np.asarray(incoming)

    def fit(self, anchor, observations):
        before,fit=anchor
        q0=np.r_[fit["pv"],fit["spin"]]
        ts=np.asarray([tick-before for tick,_ in observations],dtype=int)
        obs=np.asarray([p for _,p in observations])
        horizon=ts[-1]
        if not (2<=len(ts)<=6 and ts.min()>0 and horizon<=8):
            return None
        guess=self.generate(q0,horizon)
        if not guess[-1,-1] or not np.any(guess[-1,5]>.36) or np.linalg.norm(guess[-1,:3]-obs[-1])>.025:
            return None
        if np.count_nonzero(guess[ts-1,-1])<2:
            return None
        # Recover spin and assess its observability rather than treating a
        # tiny residual as enough. Near slip saturation most rotation is
        # nearly unobservable and the diagnostic should show that explicitly.
        use=ts-1
        def objective(x):
            q=q0+S0*x
            predicted=self.generate(q,horizon)
            loss=((predicted[use,:3]-obs)/SIGHT).ravel()
            return np.r_[loss,x]
        opt=least_squares(objective,np.zeros(9),bounds=(-2.,2.),method="trf",
                          diff_step=.006,max_nfev=12,ftol=.015,gtol=.02)
        q=q0+S0*opt.x
        pred=self.generate(q,horizon)
        if not pred[-1,-1] or pred[-1,5]<.36 or np.linalg.norm(pred[-1,:3]-obs[-1])>.0035:
            return None
        z0=objective(np.zeros(9))
        z=objective(opt.x)
        cond=np.linalg.eigvalsh(opt.jac.T@opt.jac)
        k=np.linalg.inv(opt.jac.T@opt.jac)
        return dict(pv=pred[-1,:6],spin=pred[-1,6:9],only_curvature=guess[-1,:9],
                    prior_spin=guess[-1,6:9],anchor=before,pairs=len(ts),
                    measured_cost=float(np.linalg.norm(z[:-9])),previous_cost=float(np.linalg.norm(z0[:-9])),
                    angular_ratio=np.diag(k)[6:].tolist(),condition=float(cond[-1]/cond[0]),
                    calls=opt.nfev)


class Reader:
    def __init__(self,camera,map):
        self.camera=camera
        self.map=map
        self.cal=dense.pack(camera.rig)
        self.arc=Arc(self.cal)
        self.points=[]
        self.anchors=[]

    def step(self,tick,pv):
        if not aperture(pv):
            return
        frames=self.camera.capture(pv[:3])
        pixel=camera_pixels(frames,self.camera.rig,pv)[1]
        self.arc.add(tick*DT,pixel)
        p=location(self.cal,pv,pixel)
        if p is None:
            return
        self.points.append((tick,p))
        if len(self.points)>27:
            self.points.pop(0)
        # Fit a prior only when directly observed positions descend toward
        # the near tabletop. No simulated bounce flag is used.
        if -.90<p[1]<-.16 and BOUNCE_Z-.006<p[2]<BOUNCE_Z+.28:
            if len(self.points)<2 or p[2]>=self.points[-2][1][2]-.008:
                return
            f=self.arc.fit(tick*DT,pv,spin_weight=1.,accurate=True)
            if f is not None and f["pv"][5]<-.62:
                self.anchors.append((tick,f))
                if len(self.anchors)>12:
                    self.anchors.pop(0)

    def fit(self,tick):
        if len(self.points)<3 or self.points[-1][0]!=tick:
            return None
        _,a=self.points[-2]
        _,b=self.points[-1]
        # Stereo first directly observes reversal. Further high-speed
        # observations can constrain the rebound kinematics if still timely.
        if b[2]<=a[2]+.005 or not (BOUNCE_Z-.009<a[2]<BOUNCE_Z+.21):
            return None
        for anchor in reversed(self.anchors):
            if not (2<=tick-anchor[0]<=6):
                continue
            pts=[(t,p) for t,p in self.points if t>anchor[0]]
            if any(b[0]-a[0]!=1 for a,b in zip(pts[:-1],pts[1:])):
                continue
            fit=self.map.fit(anchor,pts)
            if fit is not None:
                return fit
        return None


def main_case(ident,source,witness,waits):
    tape,chron=_read(source),_read(witness)
    row=next(x for x in chron["rows"] if x["id"]==ident)
    old=tape["rows"][row["index"]]
    ready,card=np.asarray(tape["ready"]),row["card"]
    modes=("first_pair_after","two_pairs_after","three_pairs_after")
    stops={row["fork"]+waits[k]["issued_wait_ticks"]:k for k in modes}
    with mpc.centered(ready):
        a=mpc.arena()
        low,rig,stride,lo,high=mpc.resources(a)
        lens=CroppedStereo(a,4)
        engine=mpc.OpticalClinic(body=True)
        reader=Reader(lens,BounceMap(engine.twin))
        received=[]
        ref=ready.copy()
        try:
            a.launch(old["shot"],card=card)
            a.partner.reset(seed=card["seed"])
            eye=mpc.Vision(rig,DT,raw=True)
            eye.eye=mpc.LearnedOptics(eye.eye,course.OPTIC)
            for tick in range(max(stops)+1):
                q=a.joints
                warm=tick>=row["fork"]-48
                mpc.observed(eye,tick,low.capture() if warm and tick%2==0 else None,q)
                engine.observe(tick,eye.eye,q,ref.copy())
                prior=dense.stock_at(tick*DT,eye.eye.latest)
                reader.step(tick,prior)
                if tick in stops and prior is not None and engine.sync():
                    mode=stops[tick]
                    known=waits[mode]
                    wait=known["issued_wait_ticks"]
                    if known["controls"][:wait]!=row["arms"]["stock"]["controls"][:wait]:
                        raise ValueError("Late command case changed past observations")
                    paths=[]
                    for u in (row["arms"]["stock"]["controls"][wait:],known["controls"][wait:]):
                        r=ref.copy()
                        goal=[]
                        for c in u:
                            r=np.clip(r+stride*np.asarray(c),lo,high)
                            goal.append(r.copy())
                        paths.append(mpc.tail(np.asarray(goal),mpc.HORIZON+mpc.TAIL))
                    fit=reader.fit(tick)
                    t=engine.twin
                    w=t.data.qvel[t._ball_dof+3:t._ball_dof+6]
                    nominal=engine.latest
                    phase=engine.bounced
                    def grade(z,spin,bounce):
                        engine.latest=replace(nominal,time=tick*DT,position=z[:3],velocity=z[3:6])
                        engine.bounced=bounce
                        assert engine.sync()
                        w[:]=spin
                        mujoco.mj_forward(t.model,t.data)
                        return outcome(engine,paths)|dict(state=z.tolist(),spin=spin.tolist(),
                                    ep_mm=float(np.linalg.norm(z[:3]-a.ball)*1e3),
                                    ev_mps=float(np.linalg.norm(z[3:6]-a.velocity)),
                                    ew_rps=float(np.linalg.norm(spin-a.spin)))
                    truepv=np.r_[a.ball,a.velocity]
                    ans=dict(stock=grade(prior,np.zeros(3),phase),
                             evaluator_exact_pv=grade(truepv,np.zeros(3),True),
                             evaluator_exact_pvw=grade(truepv,a.spin.copy(),True),
                             bounce=None,unchanged_curve=None)
                    if fit is not None:
                        ans["bounce"]=grade(fit["pv"],fit["spin"],True)
                        alt=fit["only_curvature"]
                        ans["unchanged_curve"]=grade(alt[:6],alt[6:9],True)
                    engine.latest,engine.bounced=nominal,phase
                    received.append(dict(tick=tick,mode=mode,actual=[False,known["landed"]],
                        after_bounce_ms=1000*known["after_event_s"],models=ans,
                        estimator=None if fit is None else {k:v for k,v in fit.items() if k not in
                                   ("pv","spin","only_curvature","prior_spin")}))
                if tick==max(stops):
                    break
                u=np.asarray(old["actions"][tick])
                ref=np.clip(ref+u*stride,lo,high)
                a.exchange(u,mpc.FAR_COMMAND)
            settled=[e for e in old["events"] if e["time"]<=a.data.time+1e-8]
            if digest_json(a.events)!=digest_json(settled):
                raise ValueError("New sensor changed native incoming trajectory")
            return dict(id=ident,phases=received)
        finally:
            engine.twin.close();lens.close();low.close();a.close()


def run(out):
    data=_read(WINDOW)
    if not data["complete"]:
        raise ValueError("The physically scored delayed alternatives are unfinished")
    result=dict(complete=False,rows=[],deadline_sha256=_hash(WINDOW),code_sha256=_hash(Path(__file__)),
                inputs="Actual 4× standard RGB after and before the bounce and the stock-achieved physical joint observations. The evaluator alone selects readout sample times and labels exact initial state diagnostic bounds. No holdout treatment issuance.")
    for ident,source,witness in CHECKS:
        old=next(r for r in data["rows"] if r["id"]==ident)
        r=main_case(ident,source,witness,old["trials"])
        result["rows"].append(r)
        result["complete"]=len(result["rows"])==len(CHECKS)
        _write(out,result)
        print(ident,[(f["mode"],f["actual"],{k:None if z is None else (z["legal"],round(z["ew_rps"],1))
                 for k,z in f["models"].items()},f["estimator"]) for f in r["phases"]],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT/"optical-bounce.json")
    run(p.parse_args().out)
