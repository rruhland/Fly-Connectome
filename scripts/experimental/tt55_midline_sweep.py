"""Probe a non-normal paddle sweep from earlier verified interior forecourt poses."""
import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from . import tt38_joint_mpc as mpc
from . import tt40_low_feasibility as native
from . import tt40_long_low as long
from . import tt51_low_reverse_curriculum as optical
from . import tt54_midline as mid
from . import tt54_midline_course as school
from . import tt55_midline_margin as margin
from . import tt55_midline_settle as settle
from .tt4_transfer_trial import _hash, _read
from .tt29_rally_physics import Rollout
from .tt37_opponent_audit import CourtClearance
from .tt38_motorcoach import AIM,answerable
from .tt44_paired_credit import State

PLAN=margin.ROOT/'sweep-protocol.json'
OUT=margin.ROOT/'sweep.json.gz'
MODES=(('down-face-fast',.50,1.55),('forward-drive',.91,1.22),('short-deep-sweep',1.40,.68))


def plan():
    d=_read(settle.OUT)
    if not d['complete']:
        raise ValueError('Cannot choose a face until actual settled contact audit completes')
    top=[]
    for group in (lambda a: a['p'][1]>-1.4 and a['face']==1,
                  lambda a:a['p'][1]>-1.4 and a['face']==-1,
                  lambda a:a['p'][1]<-1.5):
        valid=[a for a in d['attempts'] if group(a) and a['attempt'] and a['attempt']['radius_m'] is not None]
        if valid:top.append(min(valid,key=lambda s:s['attempt']['radius_m'])['route'])
    optical.save(PLAN,dict(actual_contact_audit_sha256=_hash(settle.OUT),poses=top,drives=MODES,
        lead_s=[.012,.032,.052,.074],velocity_factors=[.70,1.,1.20],time_shifts=[-.012,0.,.012,.023],
        body_earlier=11,prep_back_seconds=.079,
        rule='The two physically closest interior forecourt poses, one per measured signed face, and closest distinct deep pose. Change the earlier preparation to a bounded backswing relative to their contact poses. Deliberately sweep down the face and forward, rather than along the steep normal, with a distinct stable deep control. True ball phase is practice-only. The benchmark arm, actuator limits and actual independent opponent do not change. Retain only actual interior legal and answered responses passing full strict CAD.'))


def velocities(ik,q,face,wanted,push):
    _,axis,J,N=ik.pose(q,jacobian=True);axis=face*axis
    sweep=np.array([0.,1.,0.])-axis*axis[1]
    sweep/=max(1e-9,np.linalg.norm(sweep))
    # Maintain the desired signed normal rather than rotate into an edge.
    objective=np.r_[J,.089*N,.027*np.eye(5)]
    lower=np.where(q-ik.low<.015,-.01,-4.0)
    upper=np.where(ik.high-q<.015,+.01,4.0)
    motion=lsq_linear(objective,np.r_[axis*wanted+sweep*push,np.zeros(8)],bounds=(lower,upper)).x
    return motion,axis,J@motion


def run():
    cfg=_read(PLAN);s=_read(settle.OUT);g=_read(margin.OUT)
    if cfg['actual_contact_audit_sha256']!=_hash(settle.OUT):raise ValueError('Source contact geometry drifted')
    c0,first,ready,old=school.teacher();c=c0['card'];leg=c0['case']['leg']
    offer=old['placements']['delivered'][leg]
    source={r['route']:r for r in s['attempts']}; total=dict(plan_sha256=_hash(PLAN),rows=[],complete=False)
    fork=g['fork'];t0=g['cut']-cfg['body_earlier']
    prefix0=[e for e in old['events'] if e['time']<=t0*mpc.DT+1e-8]
    with mpc.centered(ready):
        a=mpc.arena(width=64,height=48);cad=mpc.arena(width=64,height=48,detail='cad');check=CourtClearance(cad.model,articulated=True)
        try:
            native.replay_prefix(a,old,c,fork);home=a.data.ctrl[a._arm_ids[1][2]][:5].copy()
            w=long.Workspace(a,np.asarray(ready));engine=Rollout(a,1)
            for k in cfg['poses']:
                raw=g['routed_goals'][k];q=np.asarray(raw['q']);face=raw['face'];row=source[k];t_contact=row['attempt']['placements']['delivered'][leg]['near_contact']['time']
                for name,normal,along in cfg['drives']:
                    speed,n,v=velocities(w.ik,q,face,normal,along)
                    z0=np.clip(q-cfg['prep_back_seconds']*speed,w.ik.low,w.ik.high)
                    b=np.asarray(raw['waypoint']);b=(b+.55*(z0-q)).clip(w.ik.low,w.ik.high)
                    amend=dict(raw,approach=z0,waypoint=b)
                    item=dict(route=k,face=face,type=name,actual_nominal_pose=q.tolist(),reference_skate=speed.tolist(),
                             blade_velocity_mps=v.tolist(),signed_face=n.tolist(),trials=0,legal=0,interior=0,physical=[])
                    total['rows'].append(item)
                    if not all((w.guard.clear(z0),w.guard.route(home,b,steps=13),w.guard.route(b,z0,steps=13))):
                        item['blocked_route']=True;continue
                    prefix,same,_=long.physically_prepare(a,old,c,fork,t0,amend,prefix0,brisk_flip=True)
                    item['same_physical_prefix']=same
                    if not same:continue
                    state=State(a);eta=t_contact-a.data.time
                    timeline=(np.arange(82)+1)*a.control_dt
                    options=list(itertools.product(cfg['lead_s'],cfg['velocity_factors'],cfg['time_shifts']))
                    goals=[]
                    for lead,scale,off in options:
                        span=np.clip(timeline-eta+lead+off,-cfg['prep_back_seconds'],.095)
                        g2=q+span[:,None]*speed*scale
                        goals.append(np.c_[g2,np.full(len(timeline),ready[-1])])
                    f=np.zeros((len(options),9));a.clearance_margin=.0006
                    try:
                        u,y=engine.trajectories(goals,AIM,min(.53,eta+.16),feedback=f)
                    finally:
                        a.clearance_margin=0.
                    legal=np.isfinite(y[:,0]);ok=legal&(y[:,12]<=margin.RADIUS)
                    item.update(trials=len(options),legal=int(legal.sum()),interior=int(ok.sum()),contacts=int(sum(f[:,1]>.5)),
                                interior_contacts=int(np.sum((f[:,1]>.5)&(f[:,8]<margin.RADIUS))),
                                best_forward_mps=float(max(f[:,3])))
                    quality=answerable(y)+11*y[:,12]
                    ranked=np.where(ok,quality,np.inf).argsort()[:2]
                    for j in ranked:
                        if not ok[j]:break
                        state.restore(a)
                        end=int(np.ceil((y[j,1]+.105)/a.control_dt))
                        option=dict(tape=u[j,:end].tolist(),outcome=y[j].tolist(),feedback=f[j].tolist(),params=options[j])
                        actual=margin.received(a,check,c,old,leg,prefix,option,True);item['physical'].append(actual)
                        if actual['answered'] and not actual['clearance']['penetration_samples']:
                            actual['full_cad']=mid.full_verify(ready,c,actual,check)
                            if not actual['full_cad']['penetration_samples']:
                                actual_contacts=[w for w in actual['native_contacts'] if w['side']==1]
                                if actual_contacts and actual_contacts[0]['radial_m']<=margin.RADIUS:
                                    total['accepted']=len(total['rows'])-1
                                    break
                    optical.save(OUT,total)
                    print(k,name,'blade',v.tolist(),'touched',item['contacts'],'interior contacts',item['interior_contacts'],
                          'legal/interior',item['legal'],item['interior'],'witness',total.get('accepted'),flush=True)
                    if 'accepted' in total:break
                if 'accepted' in total:break
        finally:
            a.close();cad.close()
    total['complete']=True;optical.save(OUT,total)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('mode',choices=('plan','run'))
    args=a.parse_args();plan() if args.mode=='plan' else run()
