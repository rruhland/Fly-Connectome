"""Evaluator-only interior reach and mechanical-bound census on prepared direct poses."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares

from .tt4_transfer_trial import _hash,_read,_write
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_bodycoach_windows import Reach
from .tt37_opponent import arena
from .tt40_low_feasibility import Faces
from .tt41_shallow_middle import CASES
from .tt41_shallow_model import impulse

DEPTHS=(-1.32,-1.36,-1.40,-1.44,-1.48,-1.52,-1.56)


def gap(p,center,face):
    v=p-center;axial=float(v@face)
    return axial,float(np.linalg.norm(v-axial*face))


def endpoint(kin,guard,p,v,q0,qhit,seconds,mode):
    qmin=kin.low.copy();qmax=kin.high.copy()
    _,normal=kin.pose(qhit)
    sign=1. if normal[1]>0 else -1.
    if mode=='virtual_extra_lift_elbow':
        qmin[1:3]-=.18
        qmax[1:3]+=.18
    low=np.maximum(qmin,qhit-.76);high=np.minimum(qmax,qhit+.76)
    if mode=='timed_mechanical':
        travel=4.16*seconds
        low=np.maximum(low,q0-travel);high=np.minimum(high,q0+travel)
    if np.any(low>=high):return dict(reason='empty encoder travel intersection')
    options=[]; nearest=None
    for pace,pitch in ((3.65,.19),(4.26,.05),(4.85,-.065),(3.85,.28),(4.45,.12)):
        n,_=impulse(p,v,pace,pitch)
        goal=p-.024*n
        for seed in (qhit,q0):
            current=np.clip(seed,low+1e-6,high-1e-6)
            def error(q):
                center,face=kin.pose(q)
                return np.r_[center-goal,.091*(sign*face-n),.002*(q-current)]
            def jac(q):
                y,f,L,J=kin.pose(q,jacobian=True)
                return np.r_[L,.091*sign*J,.002*np.eye(5)]
            res=least_squares(error,current,jac=jac,bounds=(low,high),max_nfev=30,ftol=1e-5,gtol=1e-6)
            q=res.x;center,face=kin.pose(q)
            axial,radius=gap(p,center,sign*face)
            alignment=float(sign*face@n)
            score=abs(axial-.024)+radius+.18*max(0.,.93-alignment)
            trial=dict(tangent_m=radius,axial_m=axial,alignment=alignment,q=q.tolist(),residual=score,
                       shoulder_margin_rad=float(q[1]-kin.low[1]),elbow_margin_rad=float(q[2]-kin.low[2]))
            if nearest is None or score<nearest['residual']:nearest=trial
            if not (.011<=axial<=.037 and radius<.120 and alignment>.925):continue
            if mode!='virtual_extra_lift_elbow' and not guard.clear(q):continue
            if mode=='timed_mechanical' and not guard.route(q0,q,steps=7):continue
            options.append(trial)
    if not options:return dict(reason='no aligned shallow-margin face',closest_residual=nearest)
    return min(options,key=lambda a:a['tangent_m'])|dict(eligible=len(options))


def analyze(output,ids):
    result=[]
    for ident in ids:
        source=CASES[ident];d=_read(source);ready=np.asarray(d['ready'])
        c,row=next((c,r) for c,r in zip(d['cards'],d['rows'],strict=True) if c['id']==ident)
        off=row['placements']['delivered'][-1];hit=off['near_contact']['time']
        bounce=off['near_bounce']['time']
        release=next(e for e in row['events'] if e['kind']=='separation' and e.get('collider') in ('table','table_right')
                     and bounce<e['time']<bounce+.043)
        fork=round((bounce-.036)*60)
        with centered(ready):
            a=arena();cad=arena(detail='cad')
            try:
                kin=Faces(cad);guard=Reach(cad,ready,ready)
                a.launch(row['shot'],card=c);a.partner.reset(seed=c['seed'])
                trace=[];state=None
                original=a._racket_contact
                def log(side,location,incoming):
                    nonlocal state
                    if side==1 and abs(a.data.time-hit)<.0025:
                        blade=a._racket_ids[1];ball=a._geom_ids['ball']
                        center=a.data.geom_xpos[blade].copy();face=a.data.geom_xmat[blade].reshape(3,3)[:,2].copy()
                        native=[r for r in a.data.contact if r.efc_address>=0 and set(r.geom)=={ball,blade}]
                        if not native:raise ValueError('Physical contact not backed by solver')
                        touch=min(native,key=lambda m:np.linalg.norm(m.pos-location))
                        h,solver=gap(touch.pos,center,face)
                        _,ray=gap(a.ball,center,face)
                        state=dict(encoder=a.joints.tolist(),center=center.tolist(),normal=face.tolist(),solver_tangent=solver,
                                   ball_center_tangent=ray,axial=h,native_contact=touch.pos.tolist())
                    return original(side,location,incoming)
                with patch.object(a,'_racket_contact',log):
                    for u in row['actions']:
                        trace.append(a.joints.copy());a.exchange(u,FAR_COMMAND)
                if state is None or digest_json(a.events)!=digest_json(row['events']):
                    raise ValueError('Verified native stock approach was not reproduced')
                q0=np.array(trace[fork])[:5]; qhit=np.asarray(state['encoder'])[:5]
                path=a.partner.flight.path(release['ball'],release['velocity'],release['spin'],1,bounced=True,horizon=.55)
                rays=[]
                for depth in DEPTHS:
                    if not len(path) or not path[:,2].min()<=depth<=path[:,2].max():continue
                    sample=path[np.argmin(abs(path[:,2]-depth))]
                    eta=release['time']+sample[0]-fork/60.
                    if eta<.05:continue
                    p=sample[1:4];v=sample[4:7]
                    modes={m:endpoint(kin,guard,p,v,q0,qhit,eta,m) for m in
                           ('timed_mechanical','untimed_mechanical','virtual_extra_lift_elbow')}
                    rays.append(dict(p=p.tolist(),v=v.tolist(),available_s=float(eta),fits=modes))
                result.append(dict(id=ident,source=source.name,source_sha256=_hash(source),fork=fork,
                    prepared=q0.tolist(),stock_contact=state,stock_limit_margins=dict(shoulder=float(qhit[1]-kin.low[1]),
                        elbow=float(qhit[2]-kin.low[2])),rays=rays,stock_owner=row['score']['learner']))
                print(ident,'actual radius',state['solver_tangent'],state['ball_center_tangent'],
                      'in-band',[[k,round(min(v['tangent_m'] for x in rays if 'tangent_m' in (v:=x['fits'][k]))*1000)] for k in
                      ('timed_mechanical','untimed_mechanical','virtual_extra_lift_elbow')
                      if any('tangent_m' in p['fits'][k] for p in rays)],flush=True)
                _write(output,dict(rows=result,depths=DEPTHS,time_proxy_rad_s=4.16,virtual_joint_extra_rad=.18,
                  method='Physical native contact radii plus local FK probe, not actual motor or an impossibility proof'))
            finally:a.close();cad.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    p.add_argument('--ids',type=int,nargs='+',default=list(CASES))
    a=p.parse_args();analyze(a.out,a.ids)
