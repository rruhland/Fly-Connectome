"""Store measured motor interfaces and recognize one from public command and encoder pairs."""
from pathlib import Path

import numpy as np

STEP=5.2/60.
PULSE=4
REST=11
SETTLE=12
AMP=.30
I=np.eye(5)


def identify(joints, step, known, coords):
    """Learn a public five-axis encoder map. The callback returns measured joints after one requested move."""
    rng=np.random.default_rng(coords)
    axes=np.linalg.qr(rng.normal(size=(5,5)))[0]
    rows=[]
    ticks=0
    for _ in range(SETTLE):
        joints=step(np.zeros(5))
        ticks+=1
    for i,direction in enumerate(axes):
        q=joints.astype(float)
        desired=AMP*direction
        for t in range(PULSE+REST):
            request=desired if t<PULSE else np.zeros(5)
            joints=step(request)
        ticks+=PULSE+REST
        rows.append((desired*STEP*PULSE,joints.astype(float)-q))
        x,y=map(np.asarray,zip(*rows,strict=True))
        residuals=[float(np.sqrt(np.mean((y-x@m.T)**2))) for m in known]
        if i==1 and known:
            best=int(np.argmin(residuals))
            if residuals[best]<.012 and sum(r<.018 for r in residuals)==1:
                return known[best],dict(ticks=ticks,known=best,residuals=residuals,x=x.tolist(),y=y.tolist())
    fit=np.linalg.solve(x,y).T
    return fit,dict(ticks=ticks,known=None,residuals=residuals,x=x.tolist(),y=y.tolist())


def sample(joints,send,seed,known=()):
    rng=np.random.default_rng(seed)
    axes=np.linalg.qr(rng.normal(size=(5,5)))[0]
    # Visit small zero-centered command displacements to avoid drift away from the sampled pose.
    points=np.r_[np.zeros((1,5)),axes*.084,np.zeros((1,5))]
    walked=[]
    spans=[]
    records=[]
    def rest(joint):
        stable=0
        for tick in range(30):
            current=send(np.zeros(5))
            stable=stable+1 if np.max(abs(current-joint))<.00045 else 0
            joint=current
            if tick>=4 and stable>=3:
                break
        return joint,tick+1
    joints,n=rest(joints)
    records.append(n)
    for goal,previous in zip(points[1:],points[:-1],strict=True):
        original=joints.astype(float)
        dx=goal-previous
        for _ in range(4):
            joints=send(dx/(STEP*4))
        joints,n=rest(joints)
        spans.append(dx)
        walked.append(joints.astype(float)-original)
        records.append(n+4)
        if len(spans)==2 and len(known):
            x,y=np.array(spans),np.array(walked)
            errors=[float(np.sqrt(np.mean((y-x@m.T)**2))) for m in known]
            best=int(np.argmin(errors))
            if errors[best]<.012 and sum(r<.018 for r in errors)==1:
                return known[best],dict(ticks=sum(records),known=best,segment_ticks=records,
                                       residuals=errors,x=x.tolist(),y=y.tolist(),fit=known[best].tolist())
    x=np.array(spans)
    y=np.array(walked)
    fit=np.linalg.lstsq(x,y,rcond=None)[0].T
    return fit,dict(ticks=sum(records),known=None,segment_ticks=records,x=x.tolist(),y=y.tolist(),fit=fit.tolist())


def decoder(mapping):
    u,s,v=np.linalg.svd(mapping)
    return (v.T*(1./np.maximum(s,.30)))@u.T


class MotorBank:
    """The supplied step callback sends five bounded controls and returns the measured joint angles."""
    def __init__(self,measured=None):
        self.maps=[I.copy()] if measured is None else [a.copy() for a in measured]
        if not self.maps or any(m.shape!=(5,5) or not np.isfinite(m).all() for m in self.maps):
            raise ValueError("expected one or more finite 5 by 5 motor maps")
        self.selected=None

    def connect(self,joints,step,seed,*,measured_settling=False):
        if measured_settling:
            mapping,report=sample(joints,step,seed,self.maps)
        else:
            mapping,report=identify(joints,step,self.maps,seed)
        self.selected=report["known"]
        if self.selected is None:
            self.selected=len(self.maps)
            self.maps.append(mapping)
        self.inverse=decoder(mapping)
        return report

    def apply(self,desired):
        if self.selected is None:
            raise RuntimeError("identify the interface before applying a learned decoder")
        return np.clip(self.inverse@desired,-1.,1.)

    def save(self,path):
        np.savez_compressed(path,schema=np.array("tt24-motor-bank-1"),maps=self.maps)

    @classmethod
    def load(cls,path):
        with np.load(Path(path),allow_pickle=False) as data:
            if data["schema"].item()!="tt24-motor-bank-1":
                raise ValueError("unknown motor bank schema")
            return cls(data["maps"])


def build():
    import gzip
    import json
    from .tt24_donor_trial import ROOT
    bank=MotorBank()
    for i in (0,1):
        with gzip.open(ROOT/f"motor-settle-{i}.json.gz","rt") as f:
            course=json.load(f)
        for session in course["sessions"][1:3]:
            bank.maps.append(np.array(session["inspection"]["fit"]))
    bank.save(ROOT/"motor-bank.npz")
    motion=MotorBank()
    for index in (0,1):
        with gzip.open(ROOT/f"motor-sampling-{index}.json.gz","rt") as f:
            source=json.load(f)
        for case in source["cases"]:
            motion.maps.append(np.array(case["adaptive"]["fit"]))
    motion.save(ROOT/"motor-bank-motion.npz")


if __name__=="__main__":
    build()
