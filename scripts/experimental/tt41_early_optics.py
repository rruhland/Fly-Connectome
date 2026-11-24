"""Causal early table-bounce estimation from the very first informative stereo frame."""
import argparse
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt33_joint as joint
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_fast_camera import native_ball_pixels
from .tt29_vision import Flight
from .tt37_visual_inertia import triangulate
from .tt37_rebound import ReboundReadout, accel, fit
from .tt37_opponent import arena
from .tt31_ready import centered

ROOT = Path("artifacts")
CHECKPOINT = ROOT / "tt37/rebound.npz"
CHECKPOINT60 = ROOT / "tt37/hz60-rebound.npz"
DEVELOP = ROOT / "tt37/hz60-practice.npz"
HELD = ROOT / "tt37/hz60-diagnosis.npz"
SIGMA = np.array([.0011,.0020,.0013])
RECEIVE = -1.12
MAX_FAST_AGE = .108


@dataclass
class Turn:
    time: float
    point: np.ndarray
    before: np.ndarray
    after: np.ndarray
    before_acceleration: np.ndarray
    after_acceleration: np.ndarray
    cost: float
    delay: float
    post_frames: int

    def state(self, now):
        dt=now-self.time
        v,a=(self.before,self.before_acceleration) if dt<0. else (self.after,self.after_acceleration)
        return np.r_[self.point+dt*v+.5*dt*dt*a,v+dt*a]


class CausalBounce:
    """Streaming optical-only piecewise fit with a learned rebound prior."""
    def __init__(self, path, *, threshold=.0042, h_sigma=.0031):
        with np.load(path) as z:
            self.incoming=z["rebound_states"].copy()
            self.delta=z["rebound_delta"].copy()
            self.height=z["rebound_z"].copy()
        self.min_height = float(np.median(self.height))
        self.threshold=threshold
        self.h_sigma=h_sigma
        self.reset()

    def reset(self):
        self.track=deque(maxlen=17)
        self.impact=None
        self.active=False
        self.last_stamp=None
        self.earliest_emission=None

    def prior(self, v):
        d2=np.sum(((self.incoming-v)/[.8,1.4,.85])**2,axis=1)
        idx=np.argpartition(d2,min(14,len(d2)-1))[:min(15,len(d2))]
        w=np.exp(-.5*d2[idx]/max(.12,float(np.median(d2[idx]))))
        w/=w.sum()
        mu=v*[1,1,-1]+w@self.delta[idx]
        sd=np.sqrt(w@((self.delta[idx]-w@self.delta[idx])**2)+np.array([.065,.12,.14])**2)
        height=float(w@self.height[idx])
        return mu,sd,height

    def candidate(self,t,p,k,now):
        # t/p are all observed through now. k is the first candidate post-impact frame.
        t0=t[k-1]; t1=t[k]
        if t1-t0 > .051 or now-t0 > MAX_FAST_AGE+.036:
            return None
        pre=t[max(0,k-6):k]-t0
        pb=p[max(0,k-6):k]
        if len(pre)<4 or pre[-1]-pre[0]<.044 or np.max(np.diff(pre))>.052:
            return None
        c,v,a=fit(pre,pb)
        if (not(-12.<v[1]<-.95 and -7.5<v[2]<-.68)
                or not(.758<c[2]<.865 and -.0>c[1]>-1.31 and abs(c[0])<.71)):
            return None
        err=np.linalg.norm(pb-c-pre[:,None]*v-.5*pre[:,None]**2*a,axis=1)
        if max(err)>.0062:
            return None
        direct=t[-1]-t0
        surprise=float(p[-1,2]-(c[2]+direct*v[2]+.5*direct**2*a[2]))
        if surprise < self.threshold + min(.0035,float(max(err))):
            return None
        post=t[k:]
        if np.any(np.diff(post)>.055) or np.any(np.diff(p[k-1:,1])>-.006):
            return None
        lo,hi=t0+.001,min(t1-.0005,now-.003)
        if hi<=lo:
            return None
        when=np.linspace(lo,hi,7)
        elapsed=when-t0
        vm=v+elapsed[:,None]*a
        # Fetch a local bounce law once for the narrow uncertainty window.
        ref,sd,height=self.prior(vm[len(when)//2])
        ve=ref+(vm-vm[len(when)//2])*[1,1,-.89]
        aa=accel(ref)
        sample=np.arange(max(0,k-6),len(t))
        dt=t[sample][None]-when[:,None]
        down=dt<0
        x=np.stack((np.ones_like(dt),np.minimum(dt,0.),np.maximum(dt,0.)),axis=-1)
        off=.5*dt[:,:,None]**2*np.where(down[:,:,None],a,aa)
        y=p[sample][None]-off
        # Solve three tiny regularized line fits simultaneously, one per stereo axis.
        n=np.einsum('vki,vkj->vij',x,x)[:,None]/SIGMA[None,:,None,None]**2
        rhs=np.einsum('vki,vkd->vdi',x,y)/SIGMA[None,:,None]**2
        n[:,2,0,0]+=1/self.h_sigma**2
        rhs[:,2,0]+=height/self.h_sigma**2
        n[:,:,2,2]+=1/sd**2
        rhs[:,:,2]+=ve/sd**2
        coeff=np.linalg.solve(n,rhs[...,None])[...,0]
        miss=p[sample][None] - off - np.einsum('vki,vdi->vkd',x,coeff)
        p0,before,after=np.moveaxis(coeff,2,0)
        goodness=np.sum((miss/SIGMA)**2,axis=(1,2))+((p0[:,2]-height)/self.h_sigma)**2
        goodness+=np.sum(((after-ve)/sd)**2,axis=1)
        ok=((before[:,2]<-.60)&(after[:,2]>.42)&(after[:,1]<-.9)&(after[:,1]/before[:,1]>.65)
            &(after[:,1]/before[:,1]<1.30)&(p0[:,2]>.751)&(p0[:,2]<.794)&(p0[:,1]>-1.37)
            &(p0[:,1]<-.04)&(abs(p0[:,0])<.735))
        good=np.flatnonzero(ok)
        if not len(good):
            return None
        i=int(good[np.argmin(goodness[good])])
        raw=np.max(np.linalg.norm(miss[i],axis=1))
        if raw>.008 or goodness[i]/(3*len(sample))>4.5:
            return None
        return Turn(float(when[i]),p0[i],before[i],after[i],a,accel(after[i]),
                    float(goodness[i]),float(now-when[i]),len(post))

    def update(self,now,point):
        self.active=False
        if self.last_stamp is not None and (now<=self.last_stamp or now-self.last_stamp>.090):
            self.reset()
        self.last_stamp=now
        if point is None or not np.isfinite(point).all():
            return None
        self.track.append((now,np.asarray(point).copy()))
        if len(self.track)<5 or not (-1.59<point[1]<-.05 and .758<point[2]<.981 and abs(point[0])<.74):
            return None
        t,p=map(np.asarray,zip(*self.track,strict=True))
        if self.impact is not None:
            if (now-self.impact.time > .128 or p[-1,1]-p[-2,1]>-.006
                    or np.linalg.norm(self.impact.state(now)[:3]-point)>.038):
                self.impact=None
                self.earliest_emission=None
        trial=[]
        for k in range(max(4,len(t)-5),len(t)):
            if self.impact is not None and abs(self.impact.time-.5*(t[k]+t[k-1]))>.030:
                continue
            r=self.candidate(t,p,k,now)
            if r is not None:
                trial.append(r)
        if trial:
            best=min(trial,key=lambda x:x.cost)
            if self.impact is None:
                self.earliest_emission=now
            self.impact=best
        if self.impact is None or not .0<now-self.impact.time<MAX_FAST_AGE:
            return None
        x=self.impact.state(now)
        if np.linalg.norm(x[:3]-point)>.014:
            return None
        self.active=True
        # A strong synchronized stereo readout corrects position without forcing its
        # frame-to-frame noise into the inferred early rebound velocity.
        x[:3]=.30*x[:3]+.70*point
        return x


def blend(now, original, early, patch_est, baseline_filter, *, released=True):
    """Switch the phase promptly but keep the trusted stock rise after it settles."""
    if patch_est is None or original is None or not released:
        return original if patch_est is None else patch_est
    turn=baseline_filter.turn
    is_rising=(turn is not None and .008<now-turn[0]<.17 and original[5]>.42
               and original[4]<-.9 and np.linalg.norm(original[:3]-patch_est[:3])<.022)
    if is_rising and (early.impact.post_frames>=2 or abs(original[5]-patch_est[5])<.25):
        return original
    return patch_est


class EarlyReadout(ReboundReadout):
    """Keep the legacy optics outside a newly verified immediate inbound rebound."""
    def __init__(self, path, **params):
        super().__init__(path)
        self.early=CausalBounce(path,**params)
        self.causal_notes=[]
        self.additional_cpu=0.

    def update(self, now, point, state):
        baseline=super().update(now,point,state)
        t0=process_time()
        estimated=self.early.update(now,point)
        answer=blend(now,baseline,self.early,estimated,self)
        self.additional_cpu+=process_time()-t0
        if self.early.active:
            turn=self.early.impact
            self.causal_notes.append(dict(now=now,impact=turn.time,post=turn.post_frames,
                                          supplied_velocity=answer is estimated))
        return answer

    def flight(self,now,frames,flight,rig):
        rays=[native_ball_pixels(frames[n]) for n in ("left","right")]
        point=None if any(r is None or r.clipped for r in rays) else triangulate([r.center for r in rays],rig)
        state=None if flight is None or now-flight.time>.07 else np.r_[*flight.advance(now)]
        choice=self.update(now,point,state)
        if choice is None or point is None:
            return flight
        if flight is None:
            damp=.133*np.clip(np.linalg.norm(choice[3:]),2.5,12.)
            return Flight(choice[:3],choice[3:],now,damp,.7,.005,self.early.impact.post_frames,True)
        return replace(flight,position=choice[:3],velocity=choice[3:],time=now,
                       bounced=flight.bounced or self.early.active)


def plane(x, now, bounced=False, damping=None, depth=RECEIVE):
    if x is None or not np.isfinite(x).all():
        return None
    d=.133*np.clip(np.linalg.norm(x[3:]),2.5,12.) if damping is None else damping
    state=Flight(x[:3],x[3:],now,d,0.,0.,4,bool(bounced))
    with patch.object(joint,"RECEIVE_Y",depth):
        return joint.flight_at_plane(state,now)


def bursts(sensor,groups,evaluator,max_age_ticks):
    # Strictly evaluator-only segmentation. Do not use this function during filtering.
    groups=np.asarray(groups)
    starts=np.r_[True,np.any(groups[1:]!=groups[:-1],axis=1)]
    out=[]
    for i in range(1,len(sensor)):
        new=(not starts[i] and evaluator[i,9]==-1 and evaluator[i,10] and evaluator[i,4]<-.85
             and not evaluator[i-1,10] and evaluator[i,1]<-.05)
        if not new:
            continue
        finish=i
        for j in range(i,min(len(sensor),i+max_age_ticks)):
            if (j>i and starts[j]) or evaluator[j,9]!=-1 or evaluator[j,4]>=-.85:
                break
            finish=j
        hit=None
        for j in range(i,finish+1):
            if evaluator[j-1,1]>=RECEIVE>evaluator[j,1]:
                a=(RECEIVE-evaluator[j-1,1])/(evaluator[j,1]-evaluator[j-1,1])
                hit=(sensor[j-1,0]+a*(sensor[j,0]-sensor[j-1,0]),
                     evaluator[j-1,:6]+a*(evaluator[j,:6]-evaluator[j-1,:6]))
                break
        out.append((i,finish,hit))
    return out


def playback(sensor,groups,path,params,*,released=True):
    f=ReboundReadout(path)
    early=CausalBounce(path,**params)
    stock=[]; alternate=[]; flag=[]; fast=[]
    cpu=[0.,0.]; used=0
    for i,(row,key) in enumerate(zip(sensor,groups,strict=True)):
        if not i or np.any(key!=groups[i-1]):
            f.last_time=None
            early.reset()
        t=row[0]; p=row[1:4]
        state=row[4:10] if np.isfinite(row[4:10]).all() and row[10]<=.07 else None
        t0=process_time()
        original=f.update(t,p,state)
        cpu[0]+=process_time()-t0
        t0=process_time()
        patch_est=early.update(t,p)
        cpu[1]+=process_time()-t0
        z=blend(t,original,early,patch_est,f,released=released)
        b=bool(row[11]) if np.isfinite(row[11]) else False
        stock.append(np.full(6,np.nan) if original is None else original)
        alternate.append(np.full(6,np.nan) if z is None else z)
        flag.append([b,b or early.active])
        fast.append(np.nan if not early.active else early.impact.time)
        used+=early.active
    return np.stack([stock,alternate]),np.asarray(flag).T,np.asarray(fast),cpu,used


def summarize(sensor,groups,evaluation,prediction,flag,turn,period):
    result={}
    cases=bursts(sensor,groups,evaluation,max_age_ticks=round(.24/period))
    offsets=(0,1,2,3) if period<.025 else (0,1,2)
    rows=[]
    t=sensor[:,0]
    for name,est,bounced in zip(("rebound","early"),prediction,flag,strict=True):
        book={}
        for j in offsets:
            ps=[]; vs=[]; fwd=[]; vz=[]; cue=[]; estim=[]
            for a,b,cross in cases:
                k=a+j
                if k>b or not np.isfinite(est[k]).all():
                    continue
                e=est[k]-evaluation[k,:6]
                ps.append(1000*np.linalg.norm(e[:3]))
                vs.append(np.linalg.norm(e[3:]))
                fwd.append(abs(e[4]))
                vz.append(abs(e[5]))
                if cross is not None and cross[0]>t[k]+.005:
                    estim.append(1)
                    state=plane(est[k],t[k],bounced[k])
                    if state is not None and state[0]>0:
                        et=1000*(t[k]+state[0]-cross[0])
                        dx=1000*(state[[1,2]]-cross[1][[0,2]])
                        cue.append([abs(et),*abs(dx)])
            book[str(j)]=dict(n=len(ps),position_mean_mm=float(np.mean(ps)) if ps else None,
                          velocity_mean_mps=float(np.mean(vs)) if vs else None,
                          forward_mae=float(np.mean(fwd)) if fwd else None,
                          vertical_mae=float(np.mean(vz)) if vz else None,
                          cue_available=len(cue),eligible_planes=len(estim),
                          cue_mae=np.mean(cue,axis=0).tolist() if cue else None)
        ready=[]; detected=[]
        for a,b,cross in cases:
            js=[k for k in range(a,b+1) if np.isfinite(est[k]).all() and est[k,4]<-.9
                and evaluation[k,5]>.23 and abs(est[k,5]-evaluation[k,5])<=.35]
            ready.append(None if not js else 1000*(t[js[0]]-t[a]))
            js=[k for k in range(a,b+1) if bounced[k]]
            detected.append(None if not js else 1000*(t[js[0]]-t[a]))
        result[name]=dict(after_first_evaluator_bounce_exposure=book,
                          usable_rise_delays_ms=ready,flag_delays_ms=detected)
    for a,b,cross in cases:
        sample=dict(card=groups[a].tolist(),first_post_s=float(t[a]),
                    actual_bounce_bracket=[float(t[a-1]),float(t[a])],actual_first_post=evaluation[a,:6].tolist(),
                    plane_cross_s=None if cross is None else float(cross[0]))
        when=next((t[k] for k in range(max(1,a-3),b+1) if np.isfinite(turn[k]) and
                   t[a-1]-.018<=turn[k]<=t[a]+.012),None)
        sample["first_early_s"]=None if when is None else float(when)
        rows.append(sample)
    return result,rows


def bench(args):
    with np.load(args.capture) as f:
        sensor,groups=f["sensor"],f["groups"]
        period=float(f["period"])
    ck=args.prior if args.prior else CHECKPOINT60 if period<.025 else CHECKPOINT
    if args.capture == HELD:
        with np.load(DEVELOP) as base:
            if np.intersect1d(groups[:,0],base["groups"][:,0]).size:
                raise ValueError("Practice and held release IDs overlap")
    pred,flag,turn,cpu,n=playback(sensor,groups,ck,dict(threshold=args.threshold,h_sigma=args.height_sigma),
                                  released=not args.full_overlay)
    # Physical evaluation is deliberately loaded only after both causal streams finish.
    with np.load(args.capture) as f:
        actual=f["evaluation"]
    result,rows=summarize(sensor,groups,actual,pred,flag,turn,period)
    false=int(np.sum(np.isfinite(turn) & ((actual[:,9]!=-1) | (actual[:,10]<.5))))
    _write(args.output,dict(schema="tt41-causal-bounce-v1",capture=str(args.capture),sha256=_hash(args.capture),
        prior=str(ck),prior_sha256=_hash(ck),threshold=args.threshold,height_sigma=args.height_sigma,
        phase_only_when_baseline_trusted=not args.full_overlay,
        period_s=period,exposures=len(sensor),active_exposures=n,false_active_exposures=false,
        stock_readout_cpu_s=cpu[0],additional_readout_cpu_s=cpu[1],results=result,opportunities=rows))
    print("period",period,"physical rebounds",len(rows),"false active",false,"CPU",cpu,flush=True)
    for key,r in result.items():
        print(key,{j:(v["n"],round(v["position_mean_mm"],2),round(v["vertical_mae"],3))
                   for j,v in r["after_first_evaluator_bounce_exposure"].items()},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--capture",type=Path,default=HELD)
    p.add_argument("--prior",type=Path)
    p.add_argument("--full-overlay",action="store_true")
    p.add_argument("--threshold",type=float,default=.0042)
    p.add_argument("--height-sigma",type=float,default=.0031)
    a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    bench(a)
