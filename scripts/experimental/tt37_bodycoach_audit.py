"""Measure the carriage-to-arm gap before checking actual coach strokes and native safety."""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
from scipy.stats import qmc

from . import tt33_joint as joint
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import BladeIK, NOMINAL_NO_LOAD
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_bodycoach import unit
from .tt37_opponent import arena
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded, verify_clearance
from .tt37_opponent_benchmark import BASE, OUT, mark, play_raw
from .tt37_opponent_capacity import ROT


def workspace(output):
    source = output/'opponent-capacity-witness.json.gz'
    row = _read(source)['rows'][0]
    ready = _read(BASE)['ready']
    plans=[r for r in row['near_plans'] if 'strike_xyz' in r]
    contacts=[c for c in row['events'] if c['kind']=='contact' and c['side']==1]
    departures=[c for c in row['events'] if c['kind']=='separation' and c['collider']=='learner_racket']
    if len(contacts)!=len(departures):
        raise ValueError('Capacity witness has an unmatched strike')
    audit={}
    for grip in ('straight','side-on'):
        with centered(ready):
            a=arena(paddle_grip=grip)
        try:
            ik=BladeIK(a)
            q=qmc.scale(qmc.Sobol(5,scramble=True,seed=371471).random_base2(12),ik.low,ik.high)
            poses=np.array([np.r_[*ik.pose(x)] for x in q])
            samples=[]
            for c,d in zip(contacts,departures,strict=True):
                goal = min(plans,key=lambda p:abs(p['time_s']+p['eta_s']-c['time']))
                normal = unit(np.asarray(d['velocity'])-c['incoming'])
                p=np.asarray(c['ball'])-.028*normal
                desired=np.asarray(goal['blade_velocity'])*ROT
                delta = (poses-np.r_[p,normal])*np.r_[1.,1.,1.,.13,.13,.13]
                order=np.argsort(np.linalg.norm(delta,axis=1))
                seeds=[np.asarray(ready)[:5]]
                for idx in order:
                    if min(np.linalg.norm(q[idx]-z) for z in seeds)>.64:
                        seeds.append(q[idx])
                    if len(seeds)>=5:
                        break
                options=[]
                for seed in seeds:
                    reach,(center,face,j,jrot)=ik.fit(p,normal,seed)
                    v=np.r_[desired,np.zeros(8)]
                    matrix=np.r_[j,.08*jrot,.01*np.eye(5)]
                    dyn=lsq_linear(matrix,v,bounds=(-NOMINAL_NO_LOAD,NOMINAL_NO_LOAD),max_iter=20).x
                    gap=float(np.linalg.norm(center-p)); aligned=float(face@normal)
                    velocity_error=float(np.linalg.norm(j@dyn-desired))
                    swing_s=float(np.max(abs(reach-np.asarray(ready)[:5]))/NOMINAL_NO_LOAD)
                    loss=gap+.20*(1-aligned)+.014*velocity_error
                    options.append((loss,dict(target_m=p.tolist(),desired_mps=desired.tolist(),normal=normal.tolist(),
                        positional_gap_m=gap,orientation_dot=aligned,velocity_error_mps=velocity_error,
                        angle_rad=reach.tolist(),nominal_ready_s=swing_s,closest_translation_mps=(j@dyn).tolist())))
                samples.append(min(options,key=lambda a:a[0])[1])
            valid=[r for r in samples if r['positional_gap_m']<=.035 and r['orientation_dot']>=.92]
            moving=[r for r in valid if r['velocity_error_mps']<=.45]
            early=[r for r in moving if r['nominal_ready_s']<=.58]
            audit[grip]=dict(total=len(samples),geometry=len(valid),geometry_and_speed=len(moving),
                             geometry_speed_and_nominal_time=len(early),samples=samples)
        finally:
            a.close()
    result=dict(capacity_sha256=_hash(source),summary=audit,comparison='Geometry-only local fitting and velocity bounds, not a physical execution or exhaustive workspace test. Normal is inferred from ball impulse.')
    _write(output/'bodycoach-workspace.json',result)


def verify(paths, output):
    results=[]
    for path in paths:
        d=_read(path)
        ready=d['ready']
        grasp=d['variant']['grasp']
        with centered(ready):
            a=arena(robust_contact=True,paddle_grip=grasp,clearance_margin=.003)
            shadow=arena(robust_contact=True,paddle_grip=grasp,clearance_margin=.003,detail='cad')
            clearance=CourtClearance(shadow.model,articulated=True)
            try:
                for c,row in zip(d['cards'],d['rows'],strict=True):
                    a.launch(row['shot'],card=c)
                    a.partner.reset(seed=c['seed'])
                    guarded(a,clearance)
                    contacts=[]
                    with contact_observer(a,contacts):
                        for x in row['actions']:
                            a.exchange(x,FAR_COMMAND)
                    actual=(a.events,a.point_placements(),audit(a.events,opening_side=1))
                    if any(digest_json(x)!=digest_json(row[k]) for x,k in zip(actual,('events','placements','score'),strict=True)):
                        raise ValueError('Coach command replay changed physical events')
                    verify_clearance(a)
                    results.append(dict(source=path.name,sha256=_hash(path),card=c,legal=row['score']['learner'],
                        solver_contacts=contacts,clearance=a.clearance_result(),events_sha256=digest_json(a.events)))
            finally:
                a.close()
                shadow.close()
    _write(output/'bodycoach-audit.json',dict(replays=results))


def baseline(path, output):
    data=_read(path)
    raw=_read(BASE)
    cap=data['variant']['cap']
    rows=[]
    with centered(raw['ready']):
        a=arena()
        renderer,rig,stride,lo,hi=joint.resources(a)
        try:
            for c,d in zip(data['cards'],data['rows'],strict=True):
                row=play_raw(a,renderer,rig,raw,stride,lo,hi,d['shot'],c,cap)
                mark(a,row,cap)
                rows.append(row)
        finally:
            renderer.close()
            a.close()
    _write(output/'bodycoach-practice-raw.json.gz',dict(rows=rows,cards=data['cards'],coach_sha256=_hash(path),
                                                       pupil_sha256=_hash(BASE),ready=raw['ready'],cap=cap))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--verify',type=Path,nargs='+')
    p.add_argument('--baseline',type=Path)
    a=p.parse_args()
    if a.verify:
        verify(a.verify,a.output)
    elif a.baseline:
        baseline(a.baseline,a.output)
    else:
        workspace(a.output)
