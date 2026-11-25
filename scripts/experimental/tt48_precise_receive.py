"""Resolve low receive states from actual dense RGB and a causal smooth-arc fit."""

import argparse
from collections import deque
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from time import process_time

import mujoco
import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_camera_precision as dense
from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_fast_camera import native_ball_pixels
from .tt29_rally_physics import BALL_RADIUS, DRAG, LIFT, SPIN_DECAY
from .tt29_vision import BOUNCE_Z, VERTICAL
from .tt42_camera_precision_report import errors
from .tt48_contact_optical_check import CHECKS
from .tt48_contact_patch import ROOT
from .tt48_optical_alternatives import outcome
from .tt48_precise_crop import Crop, rectangle

DT = 1/60
MICRO = 8
H = DT/MICRO
CAMS = ("left", "right")
ARC = .235
TIME_SCALE = .12
SPIN_SCALE = 38.
MODES = {
    "inertial": (0., False),
    "curved": (1., False),
    "curved_shape": (1., True),
}
CORPUS = Path("artifacts/tt42_camera_precision")


def aperture(pv):
    return (pv is not None and pv[4] < -.85 and 1.32 > pv[1] > -1.72
            and .72 < pv[2] < 1.83)


def camera_pixels(frame, rig, pv):
    """Measure only orange pixels in a crop centered on the causal stock estimate."""
    pix = np.full((2, 2, 2), np.nan)
    if not aperture(pv):
        return pix
    for i,k in enumerate(CAMS):
        if k not in frame:
            continue
        c = rig.fixed[k]
        projected = c.project(pv[:3])
        uv = projected[:2]
        if isinstance(frame[k],Crop):
            img,lo=frame[k].rgb,frame[k].origin
        else:
            box=rectangle(c,pv[:3])
            if box is None:
                continue
            lo,hi=box
            y,x=lo[::-1]
            v,u=hi[::-1]
            img=frame[k][y:v,x:u]
        sight=native_ball_pixels(img,uv-lo)
        # The ball often subtends <2 stock pixels on the far half, albeit
        # enough to resolve a silhouette at the upgraded sampling density.
        if sight is None or sight.clipped or sight.radius < .67*BALL_RADIUS*c.focal/projected[2]:
            continue
        center=sight.center+lo
        pix[0,i]=center
        # An unbiased silhouette has symmetric scanline chords even if
        # rasterized area fluctuates. Pair chords near the middle of each axis,
        # where the endpoint quantization projects weakly onto the chord center.
        cx,cy=np.rint(sight.center).astype(int)
        rr=int(round(1.5*sight.radius))+3
        x0,x1=max(cx-rr,0),min(cx+rr+1,img.shape[1])
        y0,y1=max(cy-rr,0),min(cy+rr+1,img.shape[0])
        patch=img[y0:y1,x0:x1].astype(np.int16)
        rgb=patch[...,0]-patch[...,2],patch[...,1]-patch[...,2]
        mask=(rgb[0]>=64)&(100*rgb[1]>25*rgb[0])&(100*rgb[1]<62*rgb[0])&(100*patch[...,2]<24*patch[...,0])
        cross=[]
        for mask1,mid,prior,off in ((mask,sight.center[1]-y0,sight.center[0]-x0,x0),
                                    (mask.T,sight.center[0]-x0,sight.center[1]-y0,y0)):
            chords=[]
            for j in range(max(0,int(round(mid-.69*sight.radius))),min(len(mask1),int(round(mid+.69*sight.radius))+1)):
                rim=np.flatnonzero(mask1[j])
                chord=0 if len(rim)==0 else rim[-1]-rim[0]+1
                if len(rim)>=.90*sight.radius and len(rim)>=.94*chord:
                    point=.5*(rim[0]+rim[-1])
                    if abs(point-prior)<.30*sight.radius:
                        weight=max(.10,1.-((j-mid)/sight.radius)**2)
                        chords.append((point+off,weight))
            if len(chords)>=4:
                pts,w=np.asarray(chords).T
                cross.append(float(w@pts/w.sum()))
            else:
                cross.append(float(prior+off))
        pix[1,i]=np.array(cross)+lo
    return pix


def backwards(pv, spin, offsets):
    """Free-flight steps, on the public sensor clock, ignoring unknown collisions."""
    v=pv[3:].copy()
    p=pv[:3].copy()
    w=spin.copy()
    ticks=np.rint(-np.asarray(offsets)/DT).astype(int)
    out=[p.copy()]
    for _ in range(int(ticks.max(initial=0))):
        for _ in range(MICRO):
            p-=H*v
            w*=1./(1.-H*SPIN_DECAY)
            # Reverse semi-implicit updates. A Taylor predictor-corrector makes
            # the drag term self-consistent to third order in one microstep.
            wx,wy,wz=w
            a=lambda z: VERTICAL-DRAG*np.sqrt(z@z)*z+LIFT*np.array(
                [wy*z[2]-wz*z[1],wz*z[0]-wx*z[2],wx*z[1]-wy*z[0]])
            pred=v-H*a(v)
            v-=H*a(pred)
        out.append(p.copy())
    return np.asarray(out)[ticks]


def skew(v):
    x,y,z=v
    return np.array([[0.,z,-y],[-z,0.,x],[y,-x,0.]])


def sphere_correct(camera, pixels, center):
    """A projected sphere's visible ellipse center is not its center's ray."""
    depth=np.einsum("nci,ci->nc", center[:,None,:]-np.asarray(camera["eyes"]),
                    np.asarray(camera["rotation"])[:,2])
    pp=np.asarray(camera["center"])
    return pp+(pixels-pp)*(1.-(BALL_RADIUS/np.clip(depth,.4,None))[:,:,None]**2)


class Arc:
    """Estimate position, velocity and the observable transverse curvature only."""
    def __init__(self, camera, *, window=ARC):
        self.camera=camera
        self.window=window
        self.samples=deque()

    def add(self, stamp, pixels):
        if np.isfinite(pixels).all():
            self.samples.append((stamp,pixels.copy()))
        while self.samples and stamp-self.samples[0][0] > self.window+1e-7:
            self.samples.popleft()

    def fit(self, now, prior, *, spin_weight=1., accurate=False):
        if not aperture(prior) or len(self.samples)<6 or now-self.samples[-1][0]>.019:
            return None
        schedule=np.asarray([t for t,_ in self.samples])-now
        # A missing pair may be tolerated, a reversal may not. The public
        # phase detector plus nested reprojection tests reject old arc samples.
        if np.max(np.diff(schedule)) > .035:
            return None
        pix=np.asarray([x for _,x in self.samples])
        center=prior[:3]+schedule[:,None]*prior[3:]+.5*schedule[:,None]**2*VERTICAL
        adjusted=sphere_correct(self.camera,pix,center)
        A0,y0=dense.matrix(self.camera,adjusted,center)
        # Real orange silhouette centroid uncertainty roughly scales with focal
        # sampling. Keep a soft physical bound on spin, not a fitted target.
        scale=self.camera["dims"][0]/384
        damp=DRAG*np.linalg.norm(prior[3:])
        t=schedule
        travel,fall=dense.flight_terms(t,damp)
        T=np.concatenate((A0,A0*travel[:,None,None,None]/TIME_SCALE),axis=-1)
        W=SPIN_SCALE*LIFT*skew(prior[3:])
        if spin_weight:
            T=np.concatenate((T,np.einsum("ncij,jk->ncik",A0,W)*fall[:,None,None,None]),axis=-1)
        free=fall[:,None]*VERTICAL
        if accurate:
            anchor=backwards(prior,np.zeros(3),t)
            affine=prior[:3]+travel[:,None]*prior[3:]+fall[:,None]*VERTICAL
            free=free+(anchor-affine)
        y=y0-np.einsum("ncij,nj->nci",A0,free)
        # Try descending suffix lengths, keep only a genuinely quiet arc.
        for cut in range(len(t)-5):
            if t[-1]-t[cut] < .081:
                break
            x,b=T[cut:].reshape(-1,T.shape[-1]),y[cut:].ravel()
            lam=np.full(x.shape[1],1e-7)
            if spin_weight:
                lam[6:]=(.30*scale/spin_weight)**2
            beta=np.linalg.solve(x.T@x+np.diag(lam),x.T@b)
            curve=free.copy()
            if spin_weight:
                curve+=fall[:,None]*(W@beta[6:])
            p,v=beta[:3],beta[3:6]/TIME_SCALE
            spin=SPIN_SCALE*beta[6:] if spin_weight else np.zeros(3)
            got=p+travel[:,None]*v+curve
            err=dense.image_residual(self.camera,got[cut:],adjusted[cut:])/scale
            if np.median(err)>.20 or np.max(err)>.70 or np.linalg.norm(p-prior[:3])>.048:
                continue
            if np.linalg.norm(v-prior[3:])>.62 or np.linalg.norm(spin)>135.:
                continue
            if p[2]<BOUNCE_Z-.006 or v[1]>-.8:
                continue
            if accurate:
                q=np.r_[p,v]
                # A second linear solve uses the true sampled gravity/drag
                # recurrence around this solution. The correction Jacobian
                # uses the local damped flight basis, whose differences decay
                # quadratically with this already small update.
                path=backwards(q,spin,t[cut:])
                dx=(y0[cut:]-np.einsum("ncij,nj->nci",A0[cut:],path)).ravel()
                ridge=np.r_[np.zeros(6), beta[6:]] if spin_weight else np.zeros(6)
                delta=np.linalg.solve(x.T@x+np.diag(lam),x.T@dx-lam*ridge)
                if np.linalg.norm(delta[:3])<.003 and np.linalg.norm(delta[3:6])/TIME_SCALE<.10:
                    beta+=delta
            return dict(pv=np.r_[beta[:3],beta[3:6]/TIME_SCALE],
                        spin=SPIN_SCALE*beta[6:] if spin_weight else np.zeros(3),
                        rms_base_px=float(np.sqrt(np.mean(err**2))),n=len(t)-cut, span=-float(t[cut]))
        return None


def corpus(path):
    meta=_read(path.with_suffix(".json"))
    with np.load(path) as z:
        rgb,stock,truth=z["rgb"][:,1],z["stock"],z["evaluator"]
        ofrec,idn=z["evaluator_offer"],z["identity"]
    variants={"inertial":(0.,False),"curve":(1.,False),"inertial_exact":(0.,True),"curve_exact":(1.,True)}
    ov={c:np.full((len(rgb),9),np.nan) for c in variants}
    stamp=idn[:,3]*DT
    for i in range(len(rgb)):
        if i==0 or not np.array_equal(idn[i,:3],idn[i-1,:3]):
            model=Arc(meta["cameras"]["high"])
        model.add(stamp[i],rgb[i])
        prior=stock[i,:6]
        # Limit this offline qualification to the optically judged receive
        # region that matters to the expensive later on-tape command test.
        if not np.isfinite(prior).all() or not aperture(prior) or prior[1] > .06:
            continue
        for c,(s,accurate) in variants.items():
            f=model.fit(stamp[i],prior,spin_weight=s,accurate=accurate)
            if f is not None:
                ov[c][i]=np.r_[f["pv"],f["spin"]]
    smooth=(truth[:,9]<0)&(truth[:,4]<-1.)&(truth[:,11]<.5)&(truth[:,1]<-.18)&(truth[:,1]>-1.65)&np.isfinite(stock).all(axis=1)
    stats=dict(releases=np.unique(idn[:,0]).astype(int).tolist())
    for part,mask in (("before",smooth&(truth[:,10]<.5)&(ofrec[:,0]-stamp<.12)&(ofrec[:,0]-stamp>.01)),
                       ("after",smooth&(truth[:,10]>.5)&(stamp-ofrec[:,0]>.034)&(stamp<ofrec[:,1]-.002)),
                       ("low_before",smooth&(truth[:,10]<.5)&(ofrec[:,0]-stamp<.12)&(ofrec[:,0]-stamp>.01)&(ofrec[:,3]<1.07))):
        page={}
        for k,p in ov.items():
            m=mask&np.isfinite(p).all(axis=1)
            pred=stock[:,:6].copy()
            pred[m]=p[m,:6]
            page[k]=dict(total=int(mask.sum()),covered=int(m.sum()),
                        position=errors(pred[mask,:3],truth[mask,:3]),
                        velocity=errors(pred[mask,3:6],truth[mask,3:6]),
                        paired_position=errors(pred[m,:3],truth[m,:3]),
                        paired_stock_position=errors(stock[m,:3],truth[m,:3]),
                        paired_velocity=errors(pred[m,3:6],truth[m,3:6]),
                        paired_stock_velocity=errors(stock[m,3:6],truth[m,3:6]),
                        derived_spin=errors(p[m,6:],truth[m,6:9]))
        stats[part]=page
    return stats


def witnessed(ident, source, path, factors):
    archive,study=_read(source),_read(path)
    if not study["complete"] or study["source_sha256"]!=_hash(source):
        raise ValueError("Changed achieved controls")
    row=next(r for r in study["rows"] if r["id"]==ident)
    old=archive["rows"][row["index"]]
    card,ready=row["card"],np.asarray(archive["ready"])
    answer=dict(id=ident, decision_tick=row["fork"],
                actual=[row["arms"][n]["landed"] for n in ("stock","edge")])
    with mpc.centered(ready):
        a=mpc.arena()
        low,rig,stride,lo,hi=mpc.resources(a)
        cameras={factor:dense.zoomed(a,factor) for factor in factors}
        models={(n,mode):Arc(dense.pack(r)) for n,(_,r) in cameras.items()
                for mode in ("area","shape")}
        engine=mpc.OpticalClinic(body=True)
        try:
            a.launch(old["shot"],card=card)
            a.partner.reset(seed=card["seed"])
            eye=mpc.Vision(rig,DT,raw=True)
            eye.eye=mpc.LearnedOptics(eye.eye,course.OPTIC)
            ref=ready.copy()
            clock=process_time()
            accepted=[]
            opened=0
            for tick in range(row["fork"]+1):
                q,now=a.joints,tick*DT
                # An unmeasured lead-in is sufficient for this isolated
                # replay. The warm interval is longer than every readout
                # and arc buffer. In live use the sensor runs continuously.
                warm=tick>=row["fork"]-48
                mpc.observed(eye,tick,low.capture() if warm and tick%2==0 else None,q)
                engine.observe(tick,eye.eye,q,ref.copy())
                prior=dense.stock_at(now,eye.eye.latest)
                if aperture(prior):
                    opened+=1
                    for n,(im,highrig) in cameras.items():
                        pixels=camera_pixels(im.capture(),highrig,prior)
                        models[n,"area"].add(now,pixels[0])
                        models[n,"shape"].add(now,pixels[1])
                if tick==row["fork"]:
                    break
                u=np.asarray(old["actions"][tick])
                ref=np.clip(ref+u*stride,lo,hi)
                a.exchange(u,mpc.FAR_COMMAND)
            recorded=[event for event in old["events"] if event["time"]<=a.data.time+1e-8]
            if digest_json(recorded)!=digest_json(a.events):
                raise ValueError("Sensor sampling changed physically achieved offer")
            if prior is None or not engine.sync():
                raise ValueError("Past stereo unexpectedly missing")
            stock=engine.latest
            goals=[]
            for name in ("stock","edge"):
                u=np.asarray(row["arms"][name]["controls"])
                v=ref.copy()
                g=[]
                for cmd in u:
                    v=np.clip(v+cmd*stride,lo,hi)
                    g.append(v)
                goals.append(mpc.tail(np.asarray(g),mpc.HORIZON+mpc.TAIL))
            twin=engine.twin
            omega=twin.data.qvel[twin._ball_dof+3:twin._ball_dof+6]
            npos=a.ball.copy()
            nvel=a.velocity.copy()
            nspin=a.spin.copy()
            # Truth is isolated to error reports and the explicitly labeled
            # p/v upper bound. The latter still has no privileged spin.
            def rank(q,spin):
                engine.latest=replace(stock,time=now,position=q[:3],velocity=q[3:6])
                assert engine.sync()
                omega[:]=spin
                mujoco.mj_forward(twin.model,twin.data)
                return outcome(engine,goals)|dict(p_error_m=float(np.linalg.norm(q[:3]-npos)),
                    v_error_mps=float(np.linalg.norm(q[3:]-nvel)),spin_error_rps=float(np.linalg.norm(spin-nspin)),
                    pv=q.tolist(),spin=spin.tolist())
            checks={"standard":rank(prior,np.zeros(3)),
                    "evaluator_exact_pv_zero_spin":rank(np.r_[npos,nvel],np.zeros(3))}
            if not factors:
                # All factor swaps belong to this lens-free evaluator mode.
                # They are never exposed to a sensor or a control policy.
                truth=(npos,nvel,nspin)
                stock_state=(prior[:3],prior[3:6],np.zeros(3))
                for n in (1,2,3):
                    for which in combinations(range(3),n):
                        p,v,w=[truth[i] if i in which else stock_state[i] for i in range(3)]
                        label="evaluator_"+"+".join(("p","v","spin")[i] for i in which)
                        checks[label]=rank(np.r_[p,v],w)
            notes={}
            for factor in factors:
                for name,(weight,shape) in MODES.items():
                    ar=models[factor,"shape" if shape else "area"]
                    fm=ar.fit(now,prior,spin_weight=weight,accurate=True)
                    tag=f"{factor}x_{name}"
                    notes[tag]=[int(round((t-now)/DT)) for t,_ in ar.samples]
                    if fm is not None:
                        accepted.append(tag)
                        checks[tag]=rank(fm["pv"],fm["spin"])|{k:v for k,v in fm.items() if k not in ("pv","spin")}
                        if weight:
                            checks[tag+"_zero_spin"]=rank(fm["pv"],np.zeros(3))
                    else:
                        checks[tag]=None
            return answer|dict(predictions=checks, accepted=accepted, opened=opened, sample_ticks=notes,
                               evaluator_only=np.r_[npos,nvel,nspin].tolist(), cpu_s=process_time()-clock)
        finally:
            engine.twin.close()
            for im,_ in cameras.values():
                im.close()
            low.close()
            a.close()


def run(cases, factors, out):
    result=dict(complete=False, hashes={str(p):_hash(p) for p in [Path(__file__),course.OPTIC]},
                rows=[],scale=factors, calibration="Public rated exposure timing, surveyed cameras, ball radius, and recorded-free-flight drag/lift. No calibrated truth fit.")
    _write(out,result)
    for case,original,witness in CHECKS:
        if case not in cases:
            continue
        result["hashes"].update({str(p):_hash(p) for p in (original,witness)})
        new=witnessed(case,original,witness,factors)
        result["rows"].append(new)
        _write(out,result)
        print(case,new["actual"],new["sample_ticks"],{k:None if f is None else (f["legal"],round(f["p_error_m"]*1e3,2),round(f["v_error_mps"],3),round(f["spin_error_rps"],1)) for k,f in new["predictions"].items()},flush=True)
    result["complete"]=len(result["rows"])==len(cases)
    _write(out,result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("accuracy","witness","exact"))
    p.add_argument("--out",type=Path)
    p.add_argument("--factors",nargs="+",type=int,default=[2,4])
    p.add_argument("--cases",nargs="+",type=int,default=[c for c,*_ in CHECKS])
    a=p.parse_args()
    if a.mode=="witness":
        run(a.cases,a.factors,a.out or ROOT/"precise-commands.json")
    elif a.mode=="exact":
        run(a.cases,[],a.out or ROOT/"precise-field-ablation.json")
    else:
        result={}
        for cert in ("practice","held"):
            source=CORPUS/(cert+".npz")
            result[cert]=corpus(source)|dict(exposures_sha256=_hash(source))
            _write(a.out or ROOT/"precise-accuracy.json",result|dict(complete=False))
            print(cert, {part: {k:[a["covered"], None if a["position"] is None else round(a["position"]["euclidean_mean"]*1e3,2),None if a["velocity"] is None else round(a["velocity"]["euclidean_mean"],3)] for k,a in t.items()} for part,t in result[cert].items() if isinstance(t,dict)},flush=True)
        result["complete"]=True
        _write(a.out or ROOT/"precise-accuracy.json",result)
