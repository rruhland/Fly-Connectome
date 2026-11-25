"""Test stable interior receiving faces, distinct from the prior moving rim strikes."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt40_long_low as long
from . import tt54_midline as mid
from . import tt54_midline_course as school
from . import tt55_midline_margin as geom
from . import tt51_low_reverse_curriculum as optical
from .tt19_view_full_game import digest_json
from .tt4_transfer_trial import _read, _hash
from .tt37_opponent_audit import CourtClearance

PLAN=geom.ROOT/'settle-protocol.json'
OUT=geom.ROOT/'settle.json.gz'


def freeze():
    existing=_read(geom.OUT); prev=_read(geom.PLAN)
    if not existing['complete']:
        raise ValueError('The prior moving approaches must conclude first')
    rows=existing['routed_goals'];minimum=sorted(rows,key=lambda z:z['quality'])
    route_indices=[next(i for i,y in enumerate(rows) if y is z) for z in minimum]
    optical.save(PLAN,dict(moving_audit_sha256=_hash(geom.OUT),route_indices=route_indices,
        earlier_settle_ticks=8,actual_target=prev['seed'],interior_radius_m=prev['interior_radius_m'],
        nearby_fresh_goals=prev['conditional_unmodified_hold'],
        role='Choose each distinct available CAD-routed face/pose from the earlier registered geometric examination. Physically reach and settle at its exact candidate contact pose eight ticks sooner than the moving arm handoff. Hold the motor reference through impact. Test all intact seed opportunities until a strictly interior legally landed and actually answered stroke is verified. Calibrating this posture is privileged practice only. The natural independent small perturbation test is reserved if an actual static interior return exists.'))


def plant(a,c,original,leg,prefix,expected,clearance=None):
    markers=[]
    if clearance is not None:
        from .tt37_opponent_audit import guarded
        guarded(a,clearance)
    else:
        a.sample_clearance(False)
    t0=a.data.time
    tape=prefix.copy()
    try:
        with geom.physical_contacts(a,markers):
            for _ in range(145):
                u=[0.]*6
                _,_,end,cap,_=a.exchange(u,mpc.FAR_COMMAND);tape.append(u)
                if end or cap or a.rival_returns>=leg+2:break
        pp=a.point_placements();o=pp['delivered'][leg] if len(pp['delivered'])>leg else None
        same=bool(o and (expected is None or all(long.same_trace(o[k],expected[k]) for k in ('far_contact','near_bounce'))))
        landed=bool(o and o['near_landed']);answered=landed and len(pp['delivered'])>leg+1
        impact=o and o['near_contact']
        radius=None
        if impact is not None:
            r=min((q for q in markers if q['side']==1),key=lambda q:abs(q['time_s']-impact['time']))
            if abs(impact['time']-r['time_s'])>1e-7:
                raise ValueError('Actually measured impact not aligned to physical trace')
            radius=r['radial_m']
        sample=a.clearance_result()
        return dict(same_offer=same,landed=landed,answered=answered,radius_m=radius,observed_native_contact=markers,
            actions=tape,shot=original['shot'],events=a.events,placements=pp,clearance=sample)
    finally:
        a.sample_clearance(False)


def run():
    p=_read(PLAN); original=_read(geom.OUT)
    if p['moving_audit_sha256']!=_hash(geom.OUT):raise ValueError('Earlier motion search changed')
    row,old_earned,ready,arch=school.teacher();c=row['card'];leg=row['case']['leg'];offer=arch['placements']['delivered'][leg]
    cut=original['cut']-p['earlier_settle_ticks'];start=original['fork'];anchor=_read(school.PLAN)['anchor']
    old_prefix=[e for e in arch['events'] if e['time']<=cut*mpc.DT+1e-8]
    data=dict(plan_sha256=_hash(PLAN),attempts=[],perturb=[],status='testing stable exact contact poses')
    with mpc.centered(ready):
        a=mpc.arena(width=64,height=48);cad=mpc.arena(width=64,height=48,detail='cad');check=CourtClearance(cad.model,articulated=True)
        try:
            witness=None
            for i in p['route_indices']:
                raw=original['routed_goals'][i]
                q=dict(raw)
                for k in ('q','approach','waypoint'):
                    q[k]=np.asarray(q[k])
                q['approach']=q['q'].copy()
                prefix,same,_=long.physically_prepare(a,arch,c,start,cut,q,old_prefix,brisk_flip=True)
                note=dict(route=i,p=q['p'],face=q['face'],prefix_matched=same,settled_q=a.joints.tolist(),attempt=None)
                if same:
                    s=plant(a,c,arch,leg,prefix,offer)
                    note['attempt']=s
                    if s['same_offer'] and s['answered'] and s['radius_m']<=p['interior_radius_m']:
                        s['full_cad']=mid.full_verify(ready,c,s,check)
                        if not s['full_cad']['penetration_samples']:
                            data.update(status='verified interior stationary pose',route=i)
                            witness=prefix
                data['attempts'].append(note);optical.save(OUT,data)
                print(i,q['p'][1],q['face'], None if note['attempt'] is None else (
                            s['same_offer'],s['landed'],s['answered'],s['radius_m']),flush=True)
                if witness:break
            if witness:
                for hold in p['nearby_fresh_goals']:
                    a.launch(arch['shot'],card=c);a.partner.reset(seed=c['seed'])
                    done=False
                    for t,u in enumerate(witness):
                        if t==anchor:a.card=dict(c,reply_seed=hold['seed'])
                        _,_,end,cap,_=a.exchange(u,mpc.FAR_COMMAND)
                        if end or cap:
                            done=True
                            break
                    if done:
                        result=dict(early_fault=True,landed=False,answered=False)
                    else:
                        result=plant(a,c,arch,leg,witness,None)
                        if result['placements']['requested'][leg]['target']!=hold['goal']:
                            raise ValueError('Actual held independent target changed')
                        if result['answered']:
                            result['full_cad']=school.independent_verify(ready,c,hold['seed'],result,check,anchor)
                    data['perturb'].append(dict(hold=hold,result=result));optical.save(OUT,data)
                    print('held',hold['seed'],result['landed'],result['answered'],result.get('radius_m'),flush=True)
            else:
                data['status']='No physically landed and answered interior static preparation'
        finally:
            a.close();cad.close()
    data['complete']=True;optical.save(OUT,data)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('mode',choices=('plan','run'))
    p=a.parse_args();freeze() if p.mode=='plan' else run()
