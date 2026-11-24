"""Acquire exact paired-camera 30 Hz rebound histories without using evaluator state in estimation."""
import argparse
from pathlib import Path

import numpy as np

from . import tt41_early_optics as track
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_vision import Rig
from .tt36_impact import triangulate

DIVERSE=track.ROOT/"tt40/long-low-diverse-optical.json.gz"
INDEPENDENT=track.ROOT/"tt38/joint-mpc-held_openings-live.json.gz"


def evaluation(a):
    # Explicitly separate truth for the audit, not a filter or selection input.
    return np.r_[a.ball,a.velocity,a.spin,a._last_hitter,a._received_bounce].copy()


def measured(imgs,rig):
    rays=[track.native_ball_pixels(imgs[s]) for s in ("left","right")]
    return None if any(v is None or v.clipped for v in rays) else triangulate([v.center for v in rays],rig)


def exposed(now,point,readout):
    return np.r_[now, np.full(3,np.nan) if point is None else point,
                 np.full(8,np.nan) if readout is None else readout]


def known(out):
    d=_read(DIVERSE)
    docs={name:_read(DIVERSE.with_name(name)) for name in d["source_sha256"]}
    keyset=set()
    sensor=[]; actual=[]; groups=[]; receipts=[]
    for g in d["groups"]:
        release=g["card"]["id"]
        leg=g["leg"]
        if (release,leg) in keyset:
            # Two polished returns of the exact same inbound launch are not two opportunities.
            continue
        keyset.add((release,leg))
        doc=docs[g["origin"]]
        row=next(r for r in doc["rows"] if r["card"]==g["card"] and r["leg"]==leg)
        win=next(r for r in row["wins"] if r["legal"] and r["answered"] and digest_json(r["events"])==g["events_sha256"])
        c=g["card"]
        t_b=win["placements"]["delivered"][leg]["near_bounce"]["time"]
        through=win["placements"]["delivered"][leg]["near_contact"]["time"]
        frames={r["tick"]:r for r in g["frames"]}
        with track.centered(doc["ready"]):
            a=track.arena()
            rig=Rig.from_arena(a)
            try:
                a.launch(win["shot"],card=c)
                a.partner.reset(seed=c["seed"])
                n=0
                for k,u in enumerate(win["actions"]):
                    t=k*a.control_dt
                    if k%2==0 and t_b-.44<=t<=through and k in frames:
                        frame=frames[k]
                        p=frame["stereo_centers_px"]
                        uv=[p.get(key) for key in ("left","right")]
                        point=triangulate(uv,rig) if all(v is not None for v in uv) else None
                        x=frame["rgb_flight"]
                        f=None if x is None else np.r_[x,0.,frame["visually_bounced"]]
                        sensor.append(exposed(t,point,f))
                        actual.append(evaluation(a))
                        groups.append([release,c["seed"],c["reply_seed"]])
                        n+=1
                    a.exchange(u,track.joint.FAR_COMMAND)
                if digest_json(a.events)!=g["events_sha256"]:
                    raise RuntimeError("Actuated rising demonstration did not replay exactly")
                receipts.append(dict(id=release,leg=leg,exposures=n,actual_bounce_s=t_b,actual_contact_s=through,
                                     events_sha256=g["events_sha256"]))
            finally:
                a.close()
    np.savez_compressed(out,sensor=sensor,evaluation=actual,groups=groups,period=1/30)
    _write(out.with_suffix(".json"),dict(source=str(DIVERSE),source_sha256=_hash(DIVERSE),rows=receipts,
                                       role="Positive low targets only, independent of causal gate tuning, not general held distribution"))
    print("Reused normal 30 Hz stereo pixels",len(receipts),len(sensor),flush=True)


def held(out,limit,source=INDEPENDENT):
    d=_read(source)
    calibration=np.load(track.DEVELOP)["groups"][:,0]
    rows=[(c,r) for c,r in zip(d["cards"],d["rows"],strict=True) if c["id"] not in calibration]
    rows=rows[:limit]
    p=[]; obs=[]; g=[]; receipts=[]
    with track.centered(d["ready"]):
        a=track.arena()
        renderer,rig,*_=track.joint.resources(a)
        try:
            for c,row in rows:
                a.launch(row["shot"],card=c)
                a.partner.reset(seed=c["seed"])
                # All first delivered physical opportunities count. Never read their outcomes during filtering.
                delivered=row["placements"]["delivered"]
                t_b=delivered[0]["near_bounce"]["time"] if delivered else None
                eye=track.joint.Vision(rig,a.control_dt,raw=True)
                n=0
                for k,u in enumerate(row["actions"]):
                    t=k*a.control_dt
                    if t_b is not None and t>t_b+.19:
                        break
                    if t_b is not None and k%2==0 and t_b-.39<=t<=t_b+.19:
                        images=renderer.capture()
                        eye.update(k,images,a.joints)
                        f=eye.eye.latest
                        x=None if f is None else np.r_[*f.advance(t),t-f.time,f.bounced]
                        p.append(exposed(t,measured(images,rig),x))
                        obs.append(evaluation(a))
                        g.append([c["id"],c["seed"],c["reply_seed"]])
                        n+=1
                    a.exchange(u,track.joint.FAR_COMMAND)
                if digest_json(a.events)!=digest_json([e for e in row["events"] if e["step"]<=a.steps]):
                    raise RuntimeError("Archived held game did not match its own referee events")
                receipts.append(dict(id=c["id"],exposures=n,bounce_s=t_b))
        finally:
            renderer.close();a.close()
    np.savez_compressed(out,sensor=p,evaluation=obs,groups=g,period=1/30)
    _write(out.with_suffix(".json"),dict(source=str(source),source_sha256=_hash(source),rows=receipts,
                                       role="First physical offers from fixed held rows, no policy intervention"))
    print("Held real 30 Hz stereo",len(receipts),len(p),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("known","held"));p.add_argument("output",type=Path)
    p.add_argument("--limit",type=int,default=8)
    p.add_argument("--source",type=Path,default=INDEPENDENT)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    known(a.output) if a.mode=="known" else held(a.output,a.limit,a.source)
