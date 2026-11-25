"""Trade some radial margin for productive orientation without changing the arm."""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from . import tt38_joint_mpc as mpc
from . import tt40_long_low as long
from . import tt40_low_feasibility as native
from . import tt51_low_reverse_curriculum as optical
from . import tt54_midline as mid
from . import tt54_midline_course as school
from . import tt55_midline_margin as margin
from . import tt55_midline_settle as settle
from .tt4_transfer_trial import _hash, _read
from .tt37_opponent_audit import CourtClearance

PLAN=margin.ROOT/'flat-protocol.json'
OUT=margin.ROOT/'flat.json.gz'
DEPTH=(-.87,-.93,-.98,-1.04,-1.10,-1.16,-1.22,-1.28,-1.67,-1.73)


def plan():
    prev=_read(settle.OUT)
    if not prev['complete']:
        raise ValueError('Physical center-contact evidence must finish before tuning slope')
    optical.save(PLAN,dict(center_contact_receipt_sha256=_hash(settle.OUT),depths=DEPTH,
        interior_fit_m=.047,interior_physical_m=.058,hypothesis='The exact center forecourt poses were too steep and wrist-flex capped. Permit a bounded center offset within the same interior face disk and prioritize flatter signed normals. Try different depths and both sides. No mechanics, opponent profile or referee changes.',
        fitter_pitch=[.18,.30,.42],static_budget=12))


def fit(ik,p,n_target,f,q_init,limit):
    def parts(q):
        b,n,J,N=ik.pose(q,jacobian=True);n=f*n;N=f*N
        d=p-b;axial=float(d@n);Dax=d@N-n@J
        rad=d-axial*n;r=float(np.linalg.norm(rad))
        Drad=rad@(-J-n[:,None]*Dax-axial*N)/max(1e-9,r)
        return axial,r,n,Dax,Drad,N
    def value(q):
        a,r,n,*_=parts(q)
        return np.r_[1.8*(a-.026),2.8*max(0.,r-limit),.12*(n-n_target),.0025*(q-q_init)]
    def derivative(q):
        a,r,n,Da,Dr,Dn=parts(q)
        return np.r_[1.8*Da[None,:],2.8*(r>limit)*Dr[None,:],.12*Dn,.0025*np.eye(5)]
    a,b=ik.low+1e-8,ik.high-1e-8
    v=least_squares(value,np.clip(q_init,a,b),jac=derivative,bounds=(a,b),max_nfev=33).x
    depth,r,n,*_=parts(v)
    return v,depth,r,n


def reserve(work,table,conf,q0,budget,old):
    track=margin.grids(work,table,conf['depths']); paths=[]; reports=[]
    for t,*r in track:
        p,v=np.array(r[:3]),np.array(r[3:6]);stats={-1:[],1:[]};min_z={-1:None,1:None}
        for target_z in conf['fitter_pitch']:
          for lateral in (-.10,0.,.10):
            n0=np.array([lateral,np.sqrt(1.-target_z**2-lateral**2),target_z])
            for f in (-1,1):
              for initial in margin.seeds(work,p,n0,f,q0,old):
                q,depth,rad,n=fit(work.ik,p,n0,f,initial,conf['interior_fit_m']-.001)
                if not .013<depth<.036 or rad>conf['interior_fit_m']+.008 or n[1]<.58:
                    continue
                if np.max(abs(q-q0))>budget*3.9 or not work.guard.clear(q):
                    continue
                min_z[f]=float(n[2]) if min_z[f] is None else min(min_z[f],float(n[2]))
                if any(abs(z['p'][1]-p[1])<.008 and z['face']==f and np.linalg.norm(z['q']-q)<.135 for z in paths):
                    continue
                center=(q0+q)*.5;lo,hi=work.ik.low,work.ik.high
                waypoint=next((w for d in ([0,0,0,0,0],[0,.33,.26,.07,0],[.16,.18,.35,-.14,0],[-.16,.18,.35,-.14,0])
                      if work.guard.route(q0,w:=np.clip(center+d,lo,hi),steps=13)
                      and work.guard.route(w,q,steps=13)),None)
                if waypoint is None:
                    continue
                rebound=v-(1.66*(v@n))*n
                time=-p[1]/max(.15,.90*rebound[1]);net=float(p[2]+time*rebound[2]-4.905*time*time)
                score=(1.5*abs(depth-.026) + .6*rad + .06*abs(n[0]) + .1*abs(n[2]-.33)
                       + .12*max(0.,.985-net) + .13*max(0.,1.6-rebound[1]))
                info=dict(time=float(t),p=p,v=v,face=f,q=q,approach=q,waypoint=waypoint,normal=n,
                          radial_m=float(rad),planar_gap_m=float(depth),score=float(score),
                          analytic_static_departure=rebound,analytic_net_height_m=net)
                paths.append(info);stats[f].append(dict(radial_m=float(rad),axial_m=float(depth),n=n.tolist()))
        reports.append(dict(depth=float(p[1]),min_normal_z=min_z,verified_routes=stats))
    paths.sort(key=lambda z:z['score']);return paths,reports


def diverse(paths,limit):
    keep=[]
    for v in paths:
        if any(v['face']==r['face'] and abs(v['p'][1]-r['p'][1])<.05 and np.linalg.norm(v['q']-r['q'])<.23 for r in keep):
            continue
        keep.append(v)
        if len(keep)==limit:break
    return keep


def run():
    c0=_read(PLAN); data=_read(settle.OUT)
    if c0['center_contact_receipt_sha256']!=_hash(settle.OUT):
        raise ValueError('Tilt-limit evidence changed')
    row,old_win,ready,old=school.teacher();c=row['card'];leg=row['case']['leg'];o=old['placements']['delivered'][leg]
    out=dict(plan_sha256=_hash(PLAN),attempts=[],perturb=[],complete=False)
    with mpc.centered(ready):
        a=mpc.arena(width=64,height=48);cad=mpc.arena(width=64,height=48,detail='cad');clear=CourtClearance(cad.model,articulated=True)
        try:
            fork,late,table=long.indexed(old['events'],o,a.control_dt); cut=late-8
            native.replay_prefix(a,old,c,fork);q0=a.data.ctrl[a._arm_ids[1][2]][:5].copy();work=long.Workspace(a,np.asarray(ready))
            early=[e for e in old['events'] if e['time']<=cut*a.control_dt+1e-8]
            first=_read(margin.OUT)['routed_goals'];seeds=[np.asarray(v['q']) for v in first if v['score']<.095][:2] if first and 'score' in first[0] else [np.asarray(v['q']) for v in first[:3]]
            paths,summary=reserve(work,table,c0,q0,(cut-fork)*a.control_dt,seeds)
            out.update(geometry=summary,supported=optical.plain(paths),start=fork,cut=cut);optical.save(OUT,out)
            print('flat geometry',len(paths),[(z['p'][1],z['face'],z['normal'][2],z['radial_m']) for z in diverse(paths,c0['static_budget'])],flush=True)
            passed=None
            for z in diverse(paths,c0['static_budget']):
                prefix,same,_=long.physically_prepare(a,old,c,fork,cut,z,early,brisk_flip=True)
                item=dict(pose=optical.plain(z),same_ordinary_prefix=same,physical=None)
                out['attempts'].append(item)
                if same:
                    got=settle.plant(a,c,old,leg,prefix,o);item['physical']=got
                    print('flat physical',float(z['p'][1]),int(z['face']),got['landed'],got['answered'],got['radius_m'],flush=True)
                    if got['answered'] and got['same_offer'] and got['radius_m']<=c0['interior_physical_m']:
                        got['full_cad']=mid.full_verify(ready,c,got,clear)
                        if not got['full_cad']['penetration_samples']:
                            passed=prefix;out['selected']=len(out['attempts'])-1
                optical.save(OUT,out)
                if passed is not None:
                    break
            if passed is not None:
                first=_read(school.PLAN)['anchor']
                for h in _read(margin.PLAN)['conditional_unmodified_hold']:
                    a.launch(old['shot'],card=c);a.partner.reset(seed=c['seed'])
                    for tick,u in enumerate(passed):
                        if tick==first:a.card=dict(c,reply_seed=h['seed'])
                        a.exchange(u,mpc.FAR_COMMAND)
                    r=settle.plant(a,c,old,leg,passed,None)
                    if r['placements']['requested'][leg]['target']!=h['goal']:
                        raise ValueError('Same-law held target mismatch')
                    if r['answered']:
                        r['full_cad']=school.independent_verify(ready,c,h['seed'],r,clear,first)
                    out['perturb'].append(dict(assigned=h,physical=r));optical.save(OUT,out)
                    print('flat held',h['seed'],r['landed'],r['answered'],r['radius_m'],flush=True)
        finally:
            a.close();cad.close()
    out.update(complete=True,status='physical interior pass' if 'selected' in out else 'no verified interior answered stroke');optical.save(OUT,out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('plan','run'))
    a=p.parse_args();plan() if a.mode=='plan' else run()
