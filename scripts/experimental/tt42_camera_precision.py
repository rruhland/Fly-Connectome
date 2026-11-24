"""Supplement ordinary RGB with denser same-pose stereo for private p/v only."""

import argparse
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path
from time import process_time

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_fast_camera import CachedStereoRenderer, native_ball_pixels
from .tt29_rally_physics import DRAG
from .tt29_vision import Rig, VERTICAL, flight_terms
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt37_opponent import arena
from .tt37_visual_inertia import LearnedOptics

ROOT = Path("artifacts")
OUT = ROOT / "tt42_camera_precision"
OPTIC = ROOT / "tt37/rebound.npz"
LIVE_ARCHIVE = ROOT / "tt39/credit-carry.json.gz"
PLAN = {
    "practice": (ROOT/"tt37/mix-rollin.json.gz", (0, 2, 3, 4)),
    "held": (ROOT/"tt39/credit-novel.json.gz", tuple(range(8))),
}
CAMS = ("left", "right")
DT = 1/60
LOOK = .135
LAM = .11


class Dimensions:
    """Only change stereo pixel sampling. Everything physical remains the same object."""
    def __init__(self, live, factor):
        self._source = live
        self.width = live.width*factor
        self.height = live.height*factor

    def __getattr__(self, key):
        return getattr(self._source,key)


def zoomed(live, factor=2):
    proxy = Dimensions(live,factor)
    return CachedStereoRenderer(proxy), Rig.from_arena(proxy)


def pack(rig):
    return dict(
        eyes=[rig.fixed[k].eye.tolist() for k in CAMS],
        rotation=[rig.fixed[k].rotation.tolist() for k in CAMS],
        focal=[rig.fixed[k].focal for k in CAMS],
        center=[rig.fixed[k].center.tolist() for k in CAMS],
        dims=[rig.fixed["left"].width,rig.fixed["left"].height],
    )


def region(pv):
    return pv is not None and pv[4] < -.92 and .86 > pv[1] > -1.72 and .67 < pv[2] < 1.72


def stock_at(now, flight):
    if (flight is None or not -.0001 <= now-flight.time < .104
            or not flight.rms<2.40 or not flight.depth_sigma < .16):
        return None
    return np.r_[*flight.advance(now)]


def bounce_flag(old, now, stock):
    """The unmodified OpticalClinic.observe rule, for paired evaluation only."""
    if stock is None or abs(now-stock.time) >= .08:
        return old
    p,v = stock.advance(now)
    if v[1] > .9:
        return False
    if v[1] < -.9 and p[1] < -.01 and (stock.bounced or v[2] > .55):
        return True
    return old


def find_spots(frame, rig, pv):
    """A wide RGB-only crop from the already available stock vision estimate."""
    q = np.full((2,2),np.nan)
    if not region(pv):
        return q
    for i,k in enumerate(CAMS):
        camera = rig.fixed[k]
        pix = camera.project(pv[:3])
        if pix[2]<=.2 or not np.isfinite(pix).all():
            return q
        uv = pix[:2]
        r = int(round(.082*camera.width))
        lim = np.array([camera.width,camera.height])
        ij = np.floor(uv).astype(int)
        lo, hi = np.maximum(ij-r,0), np.minimum(ij+r+1,lim)
        if np.any(hi-lo<12):
            continue
        x,y = lo
        u,v = hi
        sight = native_ball_pixels(frame[k][y:v,x:u],uv-lo)
        if sight is not None and not sight.clipped and sight.radius>=2.1*camera.width/384:
            q[i] = sight.center+lo
    return q


def matrix(camera, xy, prior):
    """Two rays per exposure, weighted by actual optical pixel angle and range."""
    R,E,F,C = (np.asarray(camera[key]) for key in ("rotation","eyes","focal","center"))
    # Rows are exposure-major.
    rel = (xy-C)/F[:,None]
    ray = np.concatenate((rel,np.ones(rel.shape[:-1]+(1,))),axis=-1)
    vec = np.einsum("nci,cij->ncj",ray,R)
    unit = vec/np.linalg.norm(vec,axis=-1,keepdims=True)
    proj = np.eye(3)-unit[...,None]*unit[...,None,:]
    # Re-express residuals in approximate pixels at the stock-measured range.
    depth = np.clip(np.einsum("nci,ci->nc", prior[:,None,:]-E,R[:,2]),.55,5.5)
    block = proj*(F/depth)[...,None,None]
    observed = np.einsum("ncij,cj->nci",block,E)
    return block,observed


def image_residual(camera,p,xy):
    R,E,F,C=(np.asarray(camera[key]) for key in ("rotation","eyes","focal","center"))
    projected = np.einsum("ncj,cij->nci", p[:,None,:]-E,R)
    pix = C+F[:,None]*projected[...,:2]/projected[...,2:]
    return np.linalg.norm(pix-xy,axis=-1)


class Suffix:
    """Causal motion fit to a bounded smooth suffix in their original image planes."""
    def __init__(self,camera, *, wide=False, free_force=False):
        self.camera = camera
        self.width = camera["dims"][0]
        self.wide = wide
        self.free_force=free_force
        self.samples=deque()
        self.last=None

    def update(self, now, xy, pv):
        if not region(pv):
            self.samples.clear()
            self.last=None
            return
        if np.isfinite(xy).all():
            self.samples.append((now,xy.copy()))
        while self.samples and self.samples[0][0] < now-LOOK:
            self.samples.popleft()
        if self.samples and now-self.samples[-1][0]>.045:
            self.samples.clear()
        self.last=None

    def estimate(self, now, pv, *, bounced):
        if self.last is not None and self.last[0]==now:
            return self.last[1]
        chosen = self._estimate(now,pv,bounced)
        self.last = now, chosen
        return chosen

    def _estimate(self, now, pv, bounced):
        if not bounced or not region(pv) or len(self.samples)<4:
            return None
        if now-self.samples[-1][0]>.019:
            return None
        ts = np.asarray([x[0] for x in self.samples])-now
        if np.max(np.diff(ts),initial=0.)>.045:
            return None
        pix=np.asarray([x[1] for x in self.samples])
        damp=DRAG*np.clip(np.linalg.norm(pv[3:]),2.5,9.)
        travel, fall = flight_terms(ts,damp)
        gravity = fall[:,None]*VERTICAL
        path=pv[:3]+travel[:,None]*pv[3:]+gravity
        block,target0=matrix(self.camera,pix,path)
        A=np.concatenate((block,block*travel[:,None,None,None]/LAM),axis=-1)
        if self.free_force:
            A=np.concatenate((A,block*fall[:,None,None,None]/LAM**2),axis=-1)
        target=target0-np.einsum("ncij,nj->nci",block,gravity)
        # Try only a few bounded smooth suffixes. Never borrow future images or
        # change the ball's bounce phase in the controller.
        for i in range(len(ts)-3):
            span=ts[-1]-ts[i]
            if span<.046:
                break
            if span>.103 and not self.wide:
                continue
            X=A[i:].reshape(-1,A.shape[-1])
            Y=target[i:].ravel()
            normal=X.T@X
            normal.flat[::len(normal)+1]+=1e-6
            if self.free_force:
                normal[6:,6:]+=np.eye(3)*6e4
            beta=np.linalg.solve(normal,X.T@Y)
            v=beta[3:6]/LAM
            acc=np.zeros(3) if not self.free_force else beta[6:]/LAM**2
            p=beta[:3]
            fit=p+travel[i:,None]*v+fall[i:,None]*(VERTICAL+acc)
            err=image_residual(self.camera,fit,pix[i:])*(384/self.width)
            rms=float(np.sqrt(np.mean(err**2)))
            if rms>.47 or float(np.max(err))>.90:
                continue
            if (not (-1.75<p[1]<-.32 and .768<p[2]<1.62 and v[1]<-1.0)
                    or np.linalg.norm(p-pv[:3])>.085 or np.linalg.norm(v-pv[3:])>.95
                    or (pv[5]>.49 and v[2]<-.05)):
                continue
            return dict(pv=np.r_[p,v],span_s=span,pairs=len(ts)-i,rms_base_pixel=rms)
        return None


def fuse(stock, precise, variant):
    v = {"position":0.,"hybrid":.60}.get(variant,1.)
    return np.r_[precise[:3],stock[3:]+v*(precise[3:]-stock[3:])]


class Shutter:
    """Only the optically eligible receive phase opens the extra camera."""
    def __init__(self, Camera, rig, *, variant="precise"):
        self.camera = Camera
        self.rig=rig
        self.sensor=Suffix(pack(rig),wide=variant=="wide",free_force=variant=="force")
        self.stats=Counter()
        self.notes=[]

    def begin(self):
        self.sensor.samples.clear()
        self.sensor.last=None
        self.stats.clear()
        self.notes=[]

    def step(self, now, pv):
        xyz=np.full((2,2),np.nan)
        if region(pv):
            start=process_time()
            xyz=find_spots(self.camera.capture(),self.rig,pv)
            self.stats["camera_cpu_s"]+=process_time()-start
            self.stats["high_frames"]+=1
            self.stats["stereo_pairs"]+=int(np.isfinite(xyz).all())
        self.sensor.update(now,xyz,pv)

    def accepted(self, now, pv, bounced):
        self.stats["syncs"]+=1
        result=self.sensor.estimate(now,pv,bounced=bounced)
        if result is not None:
            self.stats["high_syncs"]+=1
            self.notes.append(dict(tick=round(now/DT),delta=(result["pv"]-pv).tolist(),
                                   **{k:v for k,v in result.items() if k!="pv"}))
        return result


def installed(engine, shutter, *, variant="precise"):
    """Patch one instance only. External observations and credit features stay stock."""
    original_observe,original_sync=engine.observe,engine.sync
    def observe(tick,tracker,q,reference):
        original_observe(tick,tracker,q,reference)
        now=tick*engine.twin.control_dt
        shutter.step(now,stock_at(now,engine.latest))
    def sync():
        now=engine.tick*engine.twin.control_dt
        stock=engine.latest
        pv=stock_at(now,stock)
        update=shutter.accepted(now,pv,engine.bounced)
        if update is None:
            return original_sync()
        prediction=fuse(pv,update["pv"],variant)
        engine.latest=replace(stock,time=now,position=prediction[:3],velocity=prediction[3:])
        try:
            return original_sync()
        finally:
            engine.latest=stock
    engine.observe,engine.sync=observe,sync
    return original_observe,original_sync


def native_trace(a):
    return np.r_[a.ball,a.velocity,a.spin, float(a._last_hitter),
                 float(a._received_bounce),float(a._geom_ids["table"] in a._previous_touches)]


def offered(r,t):
    """Future event associations are written in a different, scorer-only array."""
    related=None
    for delivery in r["placements"]["delivered"]:
        if delivery["far_contact"] is None or delivery["far_contact"]["time"]>t:
            break
        related=delivery
    if related is None:
        return np.full(4,np.nan)
    start=related["near_bounce"]["time"]
    end=related["near_contact"]["time"] if related["near_contact"] is not None else np.inf
    entry=related["receive_line"]
    zz=np.nan
    if entry is not None and entry["velocity"][1]<-.9:
        eta=(-1.53-entry["xyz"][1])/entry["velocity"][1]
        zz=entry["xyz"][2]+eta*entry["velocity"][2]-4.905*eta*eta
    return np.array([start,end,float(related["near_landed"]),zz])


def save(path, chunks, memo):
    temp=path.with_name(path.name+".partial")
    with temp.open("wb") as f:
        np.savez_compressed(f,**{k:np.concatenate([x[k] for x in chunks]) for k in chunks[0]})
    temp.replace(path)
    _write(path.with_suffix(".json"),memo)


def capture(path, role, *, resume=False):
    source,indices=PLAN[role]
    spec=_read(source)
    control=_read(LIVE_ARCHIVE)
    if set(spec["cards"][i]["id"] for i in indices)&{c["id"] for c in control["cards"]}:
        raise ValueError("Sensor replay overlaps the prospective live roster")
    memo=dict(source=str(source),sha256=_hash(source),role=role,indices=indices,
              optic_sha256=_hash(OPTIC),points=[],complete=False,period=DT)
    path.parent.mkdir(parents=True,exist_ok=True)
    chunks=[]
    if resume and path.exists():
        old=_read(path.with_suffix(".json"))
        if tuple(old["indices"])!=indices or old["sha256"]!=memo["sha256"]:
            raise ValueError("Cannot resume a mismatched capture")
        memo=old
        with np.load(path) as f:
            chunks.append({k:f[k] for k in f.files})
    chosen=set(r["index"] for r in memo["points"])
    with centered(spec["ready"]):
        a=arena()
        low,rig,*_=resources(a)
        high,zoom=zoomed(a)
        memo["cameras"]={"low":pack(rig),"high":pack(zoom)}
        try:
            for idx in indices:
                if idx in chosen:
                    continue
                c,r=spec["cards"][idx],spec["rows"][idx]
                a.launch(r["shot"],card=c)
                a.partner.reset(seed=c["seed"])
                eye=Vision(rig,DT,raw=True)
                eye.eye=LearnedOptics(eye.eye,OPTIC)
                measure, truth, who, pixels, optics=[],[],[],[],[]
                phase=False
                clock=process_time()
                for tick,u in enumerate(r["actions"]):
                    now=tick*DT
                    rgb=low.capture() if tick%2==0 else None
                    eye.update(tick,rgb,a.joints)
                    f=eye.eye.latest
                    phase=bounce_flag(phase,now,f)
                    pv=stock_at(now,f)
                    l,h=np.full((2,2),np.nan),np.full((2,2),np.nan)
                    if region(pv):
                        if rgb is not None:
                            l=find_spots(rgb,rig,pv)
                        h=find_spots(high.capture(),zoom,pv)
                    optics.append(np.r_[np.full(6,np.nan) if pv is None else pv,float(phase)])
                    who.append([c["id"],c["seed"],c["reply_seed"],tick])
                    pixels.append(np.stack((l,h)))
                    measure.append(native_trace(a))
                    truth.append(offered(r,now))
                    a.exchange(u,FAR_COMMAND)
                if digest_json(a.events)!=digest_json(r["events"]):
                    raise ValueError(f"Changed native replay on {c['id']}")
                chunks.append(dict(identity=np.asarray(who),rgb=np.asarray(pixels),
                                   stock=np.asarray(optics),evaluator=np.asarray(measure),
                                   evaluator_offer=np.asarray(truth)))
                memo["points"].append(dict(index=idx,card=c,n=len(who),paired_pixels=int(np.isfinite(np.asarray(pixels)[:,1]).all(axis=(1,2)).sum()),
                                           events_sha256=digest_json(r["events"]),cpu_s=process_time()-clock))
                save(path,chunks,memo)
                print(role,c["id"],len(who),memo["points"][-1]["paired_pixels"],round(memo["points"][-1]["cpu_s"],2),flush=True)
        finally:
            high.close();low.close();a.close()
    memo["complete"]=True
    save(path,chunks,memo)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("role",choices=tuple(PLAN))
    p.add_argument("--output",type=Path)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    capture(args.output or OUT/(args.role+".npz"),args.role,resume=args.resume)
