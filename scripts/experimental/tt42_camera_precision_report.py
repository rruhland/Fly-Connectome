"""Score current and 67–100 ms free-flight error without exposing truth to the observer."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_vision import VERTICAL
from .tt42_camera_precision import DRAG, DT, OUT, Suffix, fuse, region

CANDIDATES=("precise","position","hybrid","wide","force")
ABLATIONS=("low30","high30")
HORIZONS=(4,6)
AUDIT_RANGE=(-1.65,-.48)


def propagate(pv,ticks):
    """Zero-spin private ball free-flight equation via midpoint integration."""
    p,v=pv[:,:3].copy(),pv[:,3:].copy()
    dt=DT/8
    for _ in range(ticks*8):
        a=VERTICAL-DRAG*np.linalg.norm(v,axis=1,keepdims=True)*v
        middle=v+.5*dt*a
        p+=dt*middle
        v+=dt*(VERTICAL-DRAG*np.linalg.norm(middle,axis=1,keepdims=True)*middle)
    return p


def score(data,meta):
    # The observer receives only time, two measured direction rays per frame,
    # the ordinary optical estimate and its already-issued bounce phase flag.
    t=data["identity"]
    stereo=data["rgb"]
    stock=data["stock"]
    modes={k:np.array(stock[:,:6],copy=True) for k in (*CANDIDATES,*ABLATIONS)}
    accepted={k:np.zeros(len(t),dtype=bool) for k in modes}
    spans={k:[] for k in modes}
    for i in range(len(t)):
        if i==0 or not np.array_equal(t[i,:3],t[i-1,:3]):
            fit={k:Suffix(meta["cameras"]["low" if k=="low30" else "high"],
                          wide=k=="wide",free_force=k=="force")
                 for k in ("precise","low30","high30","wide","force")}
        tick=int(t[i,-1])
        now=tick*DT
        x=stock[i,:6] if np.isfinite(stock[i,:6]).all() else None
        options={}
        for key,f in fit.items():
            pix=stereo[i,0 if key=="low30" else 1]
            if key=="high30" and tick%2:
                pix=np.full((2,2),np.nan)
            f.update(now,pix,x)
            options[key]=f.estimate(now,x,bounced=bool(stock[i,6]))
        options.update(position=options["precise"],hybrid=options["precise"])
        for key,pred in options.items():
            if pred is not None:
                modes[key][i]=fuse(x,pred["pv"],key)
                accepted[key][i]=True
                spans[key].append(pred["span_s"])
    return dict(stock=stock[:,:6],**modes),accepted,spans


def errors(x,actual):
    if not len(x):
        return None
    d=np.abs(x-actual)
    e=np.linalg.norm(d,axis=1)
    return dict(n=len(x),axis_mae=np.mean(d,axis=0).tolist(),euclidean_mean=float(e.mean()),
                euclidean_median=float(np.median(e)),euclidean_p90=float(np.quantile(e,.90)))


def paired_boot(delta, identity):
    if not len(delta):
        return None
    by=np.unique(identity[:,0])
    if len(by)<3:
        return None
    group=[delta[identity[:,0]==release].mean() for release in by]
    idx=np.random.default_rng(420981).integers(len(by),size=(1200,len(by)))
    return dict(releases=len(by),ci95=np.quantile(np.mean(np.asarray(group)[idx],axis=1),[.025,.975]).tolist())


def outcome(data,meta,pred,flags,spans, *, mode):
    identity=data["identity"]
    gt=data["evaluator"]
    offer=data["evaluator_offer"]
    stock=pred["stock"]
    n=len(gt)
    ticks=identity[:,3]
    real_time=ticks*DT
    physical=(gt[:,9]<0)&(gt[:,10]>0)&(gt[:,11]<.5)&(gt[:,4]<-.9)
    # Keep the subsequent camera-only motion estimate independent. True phase,
    # future time and referee structure first enter here in the offline scorer.
    physical &= (real_time>=offer[:,0]+.008)&(real_time<offer[:,1]-.001)
    near=(gt[:,1]>AUDIT_RANGE[0])&(gt[:,1]<AUDIT_RANGE[1])
    eligible = physical & near & np.isfinite(stock).all(axis=1)
    moderate=(offer[:,3]>=.99)&(offer[:,3]<1.11)&(offer[:,2]==0.)
    # Only score endpoints on the same uninterrupted physically free arc.
    end={}
    for k in HORIZONS:
        i=np.arange(max(n-k,0))
        j=i+k
        valid=(eligible[i]&physical[j] &
               np.all(identity[i,:3]==identity[j,:3],axis=1)&
               (ticks[j]-ticks[i]==k)&
               (offer[i,0]==offer[j,0]))
        end[k]=i[valid]
    details={}
    for tag,m in (("quiet_postbounce",eligible),("moderate_miss_postbounce",eligible&moderate)):
        d={}
        for label,forecast in pred.items():
            chosen=flags.get(label,np.zeros(n,dtype=bool))
            result=dict(position_m=errors(forecast[m,:3],gt[m,:3]),
                        velocity_mps=errors(forecast[m,3:6],gt[m,3:6]),
                        overlay=int(np.sum(chosen&m)),stock_fallback=int(np.sum(~chosen&m)),
                        groups=int(len(np.unique(identity[m,0]))),endpoint={})
            for k,i in end.items():
                i=i[m[i]]
                mm=round(1000*k*DT)
                q=propagate(forecast[i],k)
                y=gt[i+k,:3]
                old=propagate(stock[i],k)
                diff=np.linalg.norm(q-y,axis=1)-np.linalg.norm(old-y,axis=1)
                used=i[chosen[i]]
                result["endpoint"][str(mm)]=dict(full=errors(q,y),paired=errors(q[chosen[i]],y[chosen[i]]),
                    paired_stock=errors(old[chosen[i]],y[chosen[i]]),all_minus_stock=paired_boot(diff,identity[i]))
            d[label]=result
        details[tag]=d
    if mode in pred:
        d={}
        for label,m in (("quiet_postbounce",eligible),("moderate_miss_postbounce",eligible&moderate)):
            keep=m&flags[mode]
            d[label]={
                "applied":int(sum(keep)),
                "p0":errors(stock[keep,:3],gt[keep,:3]),"p1":errors(pred[mode][keep,:3],gt[keep,:3]),
                "v0":errors(stock[keep,3:6],gt[keep,3:6]),"v1":errors(pred[mode][keep,3:6],gt[keep,3:6]),
            }
        details["paired_supported"]=d
    return details


def calc(file, *, mode=None):
    meta=_read(file.with_suffix(".json"))
    with np.load(file) as d:
        pred, accepted, spans=score(d,meta)
        stats=outcome(d,meta,pred,accepted,spans,mode=mode)
    return dict(capture_sha256=_hash(file),complete=meta["complete"],source=meta["source"],sensor_ready=meta["cameras"],
                groups=[r["card"]["id"] for r in meta["points"]],statistics=stats)


def choice(practice):
    """Freeze one option using only practice endpoint error, never test/physical reward."""
    stock=practice["statistics"]["quiet_postbounce"]
    anchor=stock["stock"]
    p=anchor["position_m"]["euclidean_mean"]
    endpoints=[anchor["endpoint"][k]["full"]["euclidean_mean"] for k in ("67","100")]
    all_modes={}
    for key in CANDIDATES:
        x=stock[key]
        d=.20*x["position_m"]["euclidean_mean"]/p
        d+=.40*x["endpoint"]["67"]["full"]["euclidean_mean"]/endpoints[0]
        d+=.40*x["endpoint"]["100"]["full"]["euclidean_mean"]/endpoints[1]
        all_modes[key]=float(d)
    # Don't select extra motion flexibility on a sub-half-percent practice difference.
    best=min(all_modes.values())
    pick=next(k for k in CANDIDATES if all_modes[k]<=best+.005)
    return dict(mode=pick,practice_objective=all_modes,
                improve_on_practice=all_modes[pick]<1.)


def run(practice, held, out, *, chosen=None):
    a=calc(practice)
    freeze=choice(a) if chosen is None else chosen
    k=freeze["mode"]
    selected_train=calc(practice,mode=k)
    doc=dict(schema="tt42-precision-test-1",selection=freeze,practice=selected_train,held=None,
             note="Stock 30 Hz private phase retained; scores are factual stock physics, not used at decision time")
    if held is not None:
        doc["held"]=calc(held,mode=k)
        if set(doc["held"]["groups"])&set(doc["practice"]["groups"]):
            raise ValueError("Held releases overlap the sensor calibration group")
    _write(out,doc)
    print("selected",freeze,flush=True)
    for group in ("practice","held"):
        page=doc[group]
        if page is None:
            continue
        print(group,page["complete"],"cases",len(page["groups"]),flush=True)
        for domain in ("quiet_postbounce","moderate_miss_postbounce"):
            print(domain,flush=True)
            a=page["statistics"][domain]
            for mode in ["stock",*CANDIDATES,*ABLATIONS]:
                r=a[mode]
                val=lambda d:None if d is None else round(d["euclidean_mean"],5)
                print(mode,r["overlay"],r["stock_fallback"],val(r["position_m"]),val(r["velocity_mps"]),
                      [val(z["full"]) for z in r["endpoint"].values()],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--practice",type=Path,default=OUT/"practice.npz")
    p.add_argument("--held",type=Path)
    p.add_argument("--output",type=Path,default=OUT/"accuracy.json")
    p.add_argument("--frozen",type=Path)
    args=p.parse_args()
    run(args.practice,args.held,args.output,chosen=None if args.frozen is None else _read(args.frozen)["selection"])
