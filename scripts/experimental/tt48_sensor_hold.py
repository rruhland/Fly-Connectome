"""Measure the cropped 60 Hz sensor on predeclared, release-disjoint tapes."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_camera_precision as dense
from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt39_opportunity_map import projected_height
from .tt42_camera_precision_report import errors, paired_boot
from .tt48_contact_optical_check import CHECKS
from .tt48_contact_patch import ROOT
from .tt48_precise_crop import CroppedStereo
from .tt48_precise_receive import Arc, DT, aperture, camera_pixels


HOLD = Path("artifacts/tt42_camera_precision/held.json")


def review(entries):
    def unpack(mask, name):
        gt=np.asarray([r["evaluator"] for r in mask],dtype=float)
        at=np.asarray([r[name] if r[name] is not None else r["stock"] for r in mask],dtype=float)
        y=at-gt[:,:at.shape[1]]
        return dict(n=len(y), covered=int(sum(r[name] is not None for r in mask)),
                    p=errors(at[:,:3],gt[:,:3]),v=errors(at[:,3:6],gt[:,3:6]),
                    w=errors(at[:,6:],gt[:,6:]) if at.shape[1]>=9 else None,
                    p_lt_mm=int(np.sum(np.linalg.norm(y[:,:3],axis=1)<.001)),
                    v_lt_20mm_s=int(np.sum(np.linalg.norm(y[:,3:6],axis=1)<.020)))
    result={}
    for key in ("pre","post","low_pre"):
        mask=[e for e in entries if e["phase"]==key or (key=="low_pre" and e["phase"]=="pre" and e["height"]<1.07)]
        if not mask:
            continue
        result[key]={m:unpack(mask,m) for m in ("stock","curve","shape","inertial")}
        for m in ("curve","shape","inertial"):
            d=np.asarray([r[m] if r[m] is not None else r["stock"] for r in mask])
            q=np.asarray([r["stock"] for r in mask])
            truth=np.asarray([r["evaluator"] for r in mask])
            ident=np.asarray([r["id"] for r in mask])[:,None]
            for slot,s in (("p",slice(None,3)),("v",slice(3,6))):
                delta=np.linalg.norm(d[:,s]-truth[:,s],axis=1)-np.linalg.norm(q[:,s]-truth[:,s],axis=1)
                result[key][m][f"{slot}_paired_release_ci95"]=paired_boot(delta,ident)
    return result


def replay(a,low,rig,high,card,row):
    deliveries = [s for s in row["placements"]["delivered"] if s["near_bounce"] is not None][:2]
    a.launch(row["shot"],card=card)
    a.partner.reset(seed=card["seed"])
    eye=mpc.Vision(rig,DT,raw=True)
    eye.eye=mpc.LearnedOptics(eye.eye,course.OPTIC)
    bouts=[(s,s["near_bounce"]["time"],9.0 if s["receive_line"] is None else projected_height(s["receive_line"]),
            float("inf") if s["near_contact"] is None else s["near_contact"]["time"]) for s in deliveries]
    state=[]
    timing=dict(opened=0,sensor_cpu_s=0.,same_orange_centers=0,compared_centers=0,matched_prefix=False)
    last=-1 if not bouts else min(len(row["actions"])-1,int(np.ceil((bouts[-1][1]+.185)/DT)))
    arcs=[Arc(dense.pack(high.rig)) for _ in range(2)]
    full=None
    tested=False
    try:
        for tick in range(last+1):
            now=tick*DT
            in_view=any(-.74<now-bounce<+.183 for _,bounce,_,_ in bouts)
            image=low.capture() if in_view and tick%2==0 else None
            mpc.observed(eye,tick,image,a.joints)
            p=dense.stock_at(now,eye.eye.latest)
            if in_view and aperture(p):
                t=process_time()
                patches=high.capture(p[:3])
                pixels=camera_pixels(patches,high.rig,p)
                timing["opened"]+=1
                timing["sensor_cpu_s"]+=process_time()-t
                for k in range(2):
                    arcs[k].add(now,pixels[k])
                if not tested and np.isfinite(pixels).all() and .4>p[1]>-.6:
                    full,fullrig=dense.zoomed(a,4)
                    reference=camera_pixels(full.capture(),fullrig,p)
                    exact=np.isclose(reference,pixels,atol=1.e-10,rtol=0.)
                    timing["compared_centers"]+=int(np.isfinite(reference).all(axis=-1).sum())
                    timing["same_orange_centers"]+=int(np.all(exact,axis=-1).sum())
                    tested=True
                    full.close()
                    full=None
            for delivery,bounce,height,stop in bouts:
                ago=now-bounce
                if p is None or not -.12<ago<.173 or now>=stop-.006 or abs(ago)<.016:
                    continue
                # Score only useful leg positions, but apply no truth filter
                # until after the actual camera forecasts have been computed.
                if p[1]>-.16 or p[1]<-1.71 or p[4]>-.85:
                    continue
                guesses={}
                for k,view,spin in (("curve",0,1.),("shape",1,1.),("inertial",0,0.)):
                    est=arcs[view].fit(now,p,spin_weight=spin,accurate=True)
                    guesses[k]=None if est is None else np.r_[est["pv"],est["spin"]].tolist()
                state.append(dict(id=card["id"],tick=tick,bounce_s=bounce,height=float(height),
                                  phase="pre" if ago<0 else "post",stock=np.r_[p,np.zeros(3)].tolist(),
                                  evaluator=np.r_[a.ball,a.velocity,a.spin].tolist(),**guesses))
            a.exchange(row["actions"][tick],mpc.FAR_COMMAND)
        and_then=[e for e in row["events"] if e["time"]<=a.data.time+1.e-8]
        if same:=digest_json(a.events)==digest_json(and_then):
            timing["matched_prefix"]=same
        elif last>=0:
            raise ValueError("Sensor sampling changed native controls or referee outcomes")
        return state,timing
    finally:
        if full is not None:
            full.close()


def run(out):
    protocol=_read(HOLD)
    source=Path(protocol["source"])
    tape=_read(source)
    if protocol["sha256"]!=_hash(source):
        raise ValueError("Changed predeclared sensor holdout")
    if any(tape["cards"][i]["id"] in {k for k,*_ in CHECKS} for i in protocol["indices"]):
        raise ValueError("Sensor holdout overlaps targeted contrast")
    result=dict(complete=False,source_sha256=_hash(source),protocol_sha256=_hash(HOLD),
                end_to_end_camera_sha256=_hash(Path(__file__)), rows=[],
                scope="Actual 4× sensor tiles measured from ordinary RGB with fixed optics and a 60 Hz camera timing. Two earliest near bounces of each old tape predeclare windows; native state is only added to joint error reporting, never used as a crop, sensor phase, or fit target.")
    if out.exists():
        previous=_read(out)
        if any(previous.get(k)!=v for k,v in result.items() if k not in ("rows","complete")):
            raise ValueError("Changed sensor holdout")
        result=previous
    with mpc.centered(tape["ready"]):
        a=mpc.arena()
        low,rig,*_=mpc.resources(a)
        high=CroppedStereo(a,4)
        try:
            for i in protocol["indices"][len(result["rows"]):]:
                c,row=tape["cards"][i],tape["rows"][i]
                scored,cost=replay(a,low,rig,high,c,row)
                result["rows"].append(dict(id=c["id"],verified=cost, readouts=scored))
                result["complete"]=len(result["rows"])==len(protocol["indices"])
                result["statistics"]=review([x for row in result["rows"] for x in row["readouts"]])
                _write(out,result)
                print(c["id"],cost,len(scored),{mode:{k:[v["n"],v["covered"],
                   round(1000*v["p"]["euclidean_mean"],3),round(v["v"]["euclidean_mean"],4)]
                  for k,v in rec.items()} for mode,rec in result["statistics"].items()},flush=True)
        finally:
            high.close();low.close();a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT/"precise-held-accuracy.json.gz")
    run(p.parse_args().out)
