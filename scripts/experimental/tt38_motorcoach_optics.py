"""Causal visual estimates of whether an outgoing volley is playable by the far paddle."""

import argparse
from pathlib import Path
from collections import deque

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, resources
from .tt36_impact import acceleration_basis, estimate_impacts, rgb_point, track_at
from .tt37_opponent import arena
from .tt38_motorcoach import ARTIFACTS, OUT, SOURCE
from .tt38_motorcoach_audit import near_departure


MODEL=ARTIFACTS/'tt36'/'impact-model.json'
NET=.9325
TABLE=.78


def project(p,v,learned):
    """Forecast first bounce and contiguous useful far-paddle windows from RGB state only."""
    dt=.008
    p=np.array(p,dtype=float).copy()
    v=np.array(v,dtype=float).copy()
    bounce=None
    samples=[]
    net_height=None
    for step in range(1,230):
        accel=acceleration_basis(v)@learned
        new_v=v+dt*accel
        new_p=p+dt*(v+new_v)*.5
        if p[1]<0<=new_p[1]:
            weight=-p[1]/(new_p[1]-p[1])
            net_height=float(p[2]+weight*(new_p[2]-p[2]))
        if p[2]>=TABLE>new_p[2]:
            if bounce is not None:
                break
            w=(p[2]-TABLE)/(p[2]-new_p[2]); hit=p+w*(new_p-p)
            bounce=dict(xy=hit[:2].tolist(),time_s=(step-1+w)*dt)
            if not (0.<hit[1]<1.37 and abs(hit[0])<.765):
                break
            new_p[2]=2*TABLE-new_p[2]
            new_v*=np.array([.86,.86,-.84])
        if bounce is not None and new_v[1]>.55 and abs(new_p[0])<.60 and .66<new_p[1]<1.60 and .86<new_p[2]<1.35:
            samples.append((step*dt,*new_p.tolist()))
        p,v=new_p,new_v
        if p[1]>1.75 or abs(p[0])>.92 or p[2]<.71:
            break
    runs=[]
    for sample in samples:
        if not runs or sample[0]-runs[-1][-1][0]>dt*1.5:
            runs.append([])
        runs[-1].append(sample)
    longest=max(runs,key=len,default=[])
    return dict(bounce=bounce,net_height_m=net_height,window_s=len(longest)*dt,
                far_intercept=None if not longest else longest[len(longest)//2])


def grade(flight):
    """Smooth geometry baseline, not a calibrated score probability."""
    b=flight['bounce']
    if b is None or flight['net_height_m'] is None or flight['net_height_m']<NET:
        return 0.
    x,y=b['xy']; span=flight['window_s']
    side=np.clip((.62-abs(x))/.18,0.,1.)
    depth=np.clip((y-.15)/.32,0.,1.)*np.clip((1.45-y)/.30,0.,1.)
    return float(side*depth*np.clip(span/.11,0.,1.))


def wide_short_risk(variants):
    """Conservative escalation flag, deliberately not a rejection rule."""
    return bool(variants) and all(s['bounce'] is not None and abs(s['bounce']['xy'][0])>.55
        and s['bounce']['xy'][1]<.35 for s in variants)


def shape(track,model):
    """Classify an already-issued shot once its short visible departure packet is complete."""
    data=[(float(t),np.asarray(p),b) for t,p,b in track]
    corrections=[]
    A=np.asarray(model['acceleration'])
    buffer=deque()
    for t,p,b in data:
        buffer.append((t,p,b))
        while buffer and t-buffer[0][0]>.36:
            buffer.popleft()
        if not (-1.40 < p[1] < -.35) or len(buffer)<6:
            continue
        alarms=estimate_impacts(list(buffer),model)
        alarms=[a for a in alarms if .095<=t-a['time']<=.173 and a['outgoing'][1]>=.85 and all(
                abs(a['time']-e['time'])>.185 for e in corrections[-2:])]
        if not alarms:
            continue
        alarm=alarms[0]
        take=[(u,v) for u,v,_ in buffer if alarm['time']+.008<u<=t]
        take=take[: min(len(take),7)]
        if len(take)<4 or take[-1][0]-take[0][0]<.047:
            continue
        time=float(take[-1][0])
        dt=np.array([x-time for x,_ in take])
        p=np.array([x for _,x in take]); design=np.c_[np.ones(len(dt)),dt]
        _, v0=np.linalg.lstsq(design,p,rcond=None)[0]
        acc=acceleration_basis(v0)@A
        origin,v=np.linalg.lstsq(design,p-.5*dt[:,None]**2*acc,rcond=None)[0]
        variants=[]
        # Tolerance reflects local geometric and velocity fit noise, not exact state.
        for dx,dy,dz in ((0,0,0),(.10,.16,.14),(-.10,-.16,-.14),(.10,-.16,.14),(-.10,.16,-.14)):
            variants.append(project(origin,v+np.array([dx,dy,dz]),A))
        scores=[grade(f) for f in variants]
        corrections.append(dict(time=alarm['time'],available=float(t),window_s=time-take[0][0],
            point=origin.tolist(),velocity=v.tolist(),expected=variants[0],score=scores[0],
            low=min(scores),high=max(scores),frames=len(take),wide_short_risk=wide_short_risk(variants)))
    return corrections


def capture(paths,output):
    rows=[]
    oracle=[]
    for path in paths:
        d=_read(path)
        i=max(range(len(d['rows'])),key=lambda j:d['rows'][j]['score']['learner'])
        r,c=d['rows'][i],d['cards'][i]
        count=r['score']['learner']
        chosen=list(range(max(1,count-6),count+1))
        completed=any(e['kind']=='point' for e in r['events'])
        truth=[dict(number=k,reply_censored=k==count and not completed,**near_departure(r,k)) for k in chosen]
        # Purposeful diagnostic subset. Score emission does not read the archive or this crop definition.
        limit=truth[0]['hit']-.32
        raw=[]
        with centered(d['ready']):
            a=arena()
            renderer,rig,*_=resources(a)
            try:
                a.launch(r['shot'],card=c)
                a.partner.reset(seed=c['seed'])
                for tick,command in enumerate(r['actions']):
                    time=tick*a.control_dt
                    if time>=limit:
                        p,b=rgb_point(renderer.capture(),rig)
                        if p is not None:
                            raw.append([time,p.tolist(),None if b is None else b.tolist()])
                    a.exchange(command,FAR_COMMAND)
                if digest_json(a.events)!=digest_json(r['events']):
                    raise ValueError('RGB replay changed physical point')
            finally:
                renderer.close()
                a.close()
        rows.append(dict(tape=path.name,release=c['id'],seed=c['seed'],sha256=_hash(path),raw=raw))
        oracle.append(dict(tape=path.name,truth=truth))
        print(path.name,len(raw),flush=True)
    _write(output/'motorcoach-visual-rgb.json.gz',dict(rows=rows))
    _write(output/'motorcoach-visual-evaluator.json',dict(rows=oracle))


def evaluate(output):
    raw=_read(output/'motorcoach-visual-rgb.json.gz')['rows']
    calibrated=_read(MODEL)['models']
    prediction={hz:{r['tape']:shape(track_at(r['raw'],hz),calibrated[str(hz)]) for r in raw} for hz in (30,60)}
    truth=_read(output/'motorcoach-visual-evaluator.json')['rows']
    result={}
    for hz,detect in prediction.items():
        points=[]
        for r in truth:
            for row in r['truth']:
                options=[p for p in detect[r['tape']] if abs(p['time']-row['hit'])<=.072]
                best=min(options,key=lambda x:abs(x['time']-row['hit'])) if options else None
                points.append(dict(tape=r['tape'],actual=row,visual=best))
        scored=[r for r in points if r['visual'] is not None and r['visual']['expected']['bounce'] is not None]
        responders=[p for p in scored if p['actual']['far_answered']]
        no_reply=[p for p in scored if not p['actual']['far_answered'] and not p['actual'].get('reply_censored')]
        err=[np.subtract(e['visual']['expected']['bounce']['xy'],e['actual']['landing'][:2]) for e in scored]
        latency=[p['visual']['available']-p['actual']['hit'] for p in scored]
        result[str(hz)]=dict(points=points,coverage=[len(scored),len(points)],
            median_xy_error_m=np.median(abs(np.asarray(err)),axis=0).tolist() if err else None,
            median_delay_s=float(np.median(latency)) if latency else None,
            no_reply_risk=[sum(p['visual']['wide_short_risk'] for p in no_reply),len(no_reply)],
            reply_risk=[sum(p['visual']['wide_short_risk'] for p in responders),len(responders)],
            answered_grade_zero=sum(p['visual']['score']==0. for p in responders))
    _write(output/'motorcoach-visual-results.json',dict(results=result,calibration_sha256=_hash(MODEL)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--capture',type=Path,nargs='*')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.capture is not None:
        capture(a.capture or [SOURCE,OUT/'motorcoach-suffix-clinic.json.gz',OUT/'motorcoach-suffix-early.json.gz'],a.output)
    evaluate(a.output)
