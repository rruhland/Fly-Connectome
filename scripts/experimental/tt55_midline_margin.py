"""Find a different, interior-impact forecourt family on intact central-low physics."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from . import tt38_joint_mpc as mpc
from . import tt40_long_low as long
from . import tt40_low_feasibility as native
from . import tt54_midline as mid
from . import tt54_midline_course as school
from . import tt51_low_reverse_curriculum as optical
from .tt4_transfer_trial import _hash, _read
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import Rollout, desired_shot
from .tt37_motor_clinic import landing_error
from .tt37_opponent_audit import CourtClearance, guarded, contact_observer
from .tt38_motorcoach import AIM, SIZES, answerable, schedules
from .tt44_paired_credit import State

ROOT=Path('artifacts/tt55_midline_margin')
PLAN=ROOT/'plan.json'
OUT=ROOT/'result.json.gz'
DEPTH=(-1.00,-1.07,-1.13,-1.20,-1.27,-1.34,-1.41,-1.48,-1.55,-1.62,-1.69,-1.76,-1.83,-1.90)
RADIUS=.058


def physical_contacts(a,out):
    start=a._racket_contact

    def contact(side,point,incoming):
        own=a._racket_ids[side];ball=a._geom_ids['ball']
        matches=[c for c in a.data.contact if c.efc_address>=0 and set(c.geom)=={ball,own}]
        if not matches:
            raise ValueError('Cannot measure a radius without its actual contact')
        c=min(matches,key=lambda c:np.linalg.norm(c.pos-point))
        a0=a.data.geom_xmat[own].reshape(3,3)[:,2]
        offset=c.pos-a.data.geom_xpos[own]
        r=np.linalg.norm(offset-a0*(offset@a0))
        orientation=c.frame[:3]*(1 if int(c.geom[0])==own else -1)
        out.append(dict(side=side,time_s=float(a.data.time),radial_m=float(r),
                        face_alignment=float(a0@orientation),racket=a.data.geom_xpos[own].tolist()))
        return start(side,point,incoming)

    return patch.object(a,'_racket_contact',contact)


def audit_finite():
    p=_read(OUT); row,teacher,ready,old=school.teacher();c=row['card'];leg=row['case']['leg']
    refs=[('previous rim',teacher)]
    refs.extend((f'new route-{z["index"]}',t) for z in p['routes'] for t in z['physically_tested'] if t['answered'])
    result=[]
    with mpc.centered(ready):
        a=mpc.arena(width=64,height=48)
        try:
            for name,v in refs:
                a.launch(v['shot'],card=c);a.partner.reset(seed=c['seed'])
                measured=[]
                with physical_contacts(a,measured):
                    for u in v['actions']:
                        a.exchange(u,mpc.FAR_COMMAND)
                if digest_json(a.events)!=digest_json(v['events']):
                    raise ValueError('Physical contact geometry replay changed')
                when=v['placements']['delivered'][leg]['near_contact']['time']
                m=min([z for z in measured if z['side']==1],key=lambda z:abs(z['time_s']-when))
                if abs(m['time_s']-when)>1e-7:
                    raise ValueError('Actual witness contact is not the claimed event')
                result.append(dict(name=name,actual=m))
        finally:
            a.close()
    optical.save(ROOT/'contact-geometry.json',dict(source_sha256=_hash(OUT),rows=result))


def freeze():
    row,teacher,ready,old=school.teacher(); prior=_read(school.PLAN)
    c=row['card']; origin=np.array(prior['seed_target']);rng=np.random.default_rng(550108)
    held=[]
    while len(held)<4:
        seed=int(rng.integers(1,2**30));g=school.STD(dict(c,reply_seed=seed),prior['leg'],'random',None)[0]
        if abs(g[0]-origin[0])<.035 and abs(g[2]-origin[2])<.021 and abs(g[3]-origin[3])<.26:
            held.append(dict(seed=seed,goal=list(g)))
    ROOT.mkdir(parents=True,exist_ok=True)
    optical.save(PLAN,dict(midline_protocol_sha256=_hash(school.PLAN),seed=c['reply_seed'],leg=prior['leg'],
         depths=DEPTH,maximum_routes=6,native_batch=144,rounds=3,interior_radius_m=RADIUS,
         alternate_shadow_limit=-1.97,conditional_unmodified_hold=held,
         rule='Recover the same archived signed prior strike. Multi-start nonlinear contact geometry in both physical faces with varied tilt and optical-independent local context. Require virtual contact radius <=.058m before a candidate is considered interior. Early fixed posture/route is physically issued. Physically reproduce actual target and offer, verify entire CAD tape. Do not run held same-law goals unless one such physical return occurs. Privileged during mechanism search only.'))


def grids(work,table,depths):
    path=long.defensive_flight(work.a.partner.flight,table)
    return [path[np.argmin(abs(path[:,2]-v))] for v in depths if path[:,2].min()-.015<v<path[:,2].max()+.015]


def seeds(work,p,n,face,q0,old):
    aim=np.r_[p-.025*n,face*n]/long.SCALE
    _,j=work.index.query(aim,k=18)
    near=work.poses[j]; key=near[0]
    different=near[np.argmax(np.linalg.norm(near-key,axis=1))]
    return [q0, key, different, *old]


def candidate_geometry(work,track,q0,budget,old):
    result=[]; rows=[]
    ik=work.ik
    for t,*rest in track:
        p,v=np.array(rest[:3]),np.array(rest[3:6])
        best={-1:None,1:None}; count={-1:0,1:0}
        for pace in (3.6,4.7,5.5):
          nominal=desired_shot(p,v,1,[0.,.65],pace)
          for pitch,yaw in ((-.27,-.10),(-.12,.10),(0.,0.),(.16,-.09),(.28,.10)):
            desire=nominal+[yaw,0.,pitch]; desire/=np.linalg.norm(desire)
            # Keep the remaining geometry permissive to finite contact
            # velocity variations, but optimize against the actual unshifted
            # ball after taking the planar blade thickness into account.
            for f in (-1,1):
              for q_init in seeds(work,p,desire,f,q0,old):
                lo,hi=ik.low+1e-8,ik.high-1e-8
                if np.linalg.norm(q_init-q0)>.60+4.16*budget*np.sqrt(5):
                    continue
                for orientation in (.036,.080):
                  def resid(q):
                    pos,axis=ik.pose(q);n=f*axis
                    d=p-pos
                    return np.r_[d-.026*n,orientation*(n-desire),.004*(q-q_init)]
                  def jac(q):
                    pos,axis,J,N=ik.pose(q,jacobian=True)
                    return np.r_[-J-f*.026*N,orientation*f*N,.004*np.eye(5)]
                  q=least_squares(resid,np.clip(q_init,lo,hi),jac=jac,bounds=(lo,hi),max_nfev=26).x
                  b,a,j,nJ=ik.pose(q,jacobian=True); a=a*f; nJ=nJ*f
                  d=p-b; axial=float(d@a); radial=float(np.linalg.norm(d-axial*a)); align=desire@a
                  grade=radial+1.7*abs(axial-.026)+.08*(1.-align)
                  summary=dict(radial_m=radial,axial_m=axial,alignment=float(align),grade=float(grade),face=f,
                               q=q.tolist(),p=p.tolist(),depth=float(p[1]),pace=pace)
                  if best[f] is None or grade<best[f]['grade']:
                    best[f]=summary
                  if (radial>.063 or not .004<axial<.051 or a[1]<.46 or align<.64
                          or np.max(abs(q-q0))>budget*3.9 or not work.guard.clear(q)):
                    continue
                  if any(abs(z['p'][1]-p[1])<.008 and z['face']==f and np.linalg.norm(z['q']-q)<.24 for z in result):
                    continue
                  dq=np.linalg.lstsq(np.r_[j,.050*nJ,.032*np.eye(5)],
                         np.r_[max(.46,(pace+v[1])*.40)*a+[0.,.10,.30],np.zeros(8)],rcond=None)[0].clip(-5.0,5.0)
                  approach=np.clip(q-.052*dq,lo,hi)
                  if not work.guard.clear(approach):
                    approach=q.copy()
                  middle=(q0+approach)*.5
                  waypoint=next((w for b0 in ([0,0,0,0,0],[0,.35,.21,.09,0],[.16,.20,.36,-.14,0],[-.16,.20,.36,-.14,0])
                     if work.guard.route(q0,w:=np.clip(middle+b0,lo,hi),steps=13)
                     and work.guard.route(w,approach,steps=13)),None)
                  if waypoint is None:
                    continue
                  entry=dict(summary,time=float(t),v=v,normal=a,n=a,q=q,dq=dq,approach=approach,waypoint=waypoint,
                             quality=grade+.007*np.linalg.norm(q-q0))
                  result.append(entry);count[f]+=1
        rows.append(dict(depth=float(p[1]),closest=best,route_counts=count))
    return result,rows


def route_selection(reachable,budget):
    # Forecourt first, but preserve a distinct mid/deep or opposite-face
    # fallback when the previous geometry family cannot meet the margin.
    groups=[]
    for y0,y1 in ((-1.35,-.90),(-1.64,-1.35),(-2.0,-1.64)):
        for f in (1,-1):
            r=sorted([z for z in reachable if y0<=z['p'][1]<y1 and f==z['face']],key=lambda z:z['quality'])
            if r:groups.append(r)
    result=[x[0] for x in groups]
    for x in groups:
        result.extend(z for z in x[1:3] if all(np.linalg.norm(z['q']-k['q'])>.23 for k in result))
    return result[:budget]


def physical_model(a,engine,anchor,neighbors,cut_time,batch,rounds,seed):
    near=[z for z in neighbors if z['face']==anchor['face'] and np.linalg.norm(z['q']-anchor['q'])<.72
                                              and abs(z['p'][1]-anchor['p'][1])<.23]
    near=sorted(near,key=lambda z:z['quality']+.12*np.linalg.norm(z['q']-a.joints[:5]))[:15]
    if not near:
        return [],[]
    eta=np.array([z['time']+cut_time-a.data.time for z in near]);q=np.array([z['q'] for z in near]);w=np.array([z['dq'] for z in near])
    rng=np.random.default_rng(seed); n=round((max(eta)+1.04)/a.control_dt)
    if n<40:return [],[]
    scales=SIZES*np.array([.80,.74,.78,.80,.79,.74,.80,.71,.78,.86,.70,.65,.65])
    raw=qmc.Sobol(len(scales),scramble=True,seed=int(rng.integers(1,2**31))).random_base2(int(np.ceil(np.log2(batch))))[:batch]
    pars=(2*raw-1)*scales; ids=np.arange(batch)%len(near)
    pars[:len(near)]=0.
    archive=[]; result=[]; log=[]
    for turn in range(rounds):
        goals=schedules(q[ids],w[ids],eta[ids],pars,a.control_dt,n)
        ff=np.zeros((batch,9));u,y=engine.trajectories(goals,AIM,max(.20,float(max(eta)+.13)),feedback=ff)
        legal=np.isfinite(y[:,0]);contact=ff[:,1]>.5
        interior=legal & (y[:,12]<=RADIUS)
        loss=np.where(legal,answerable(y)-14.+38.*np.maximum(y[:,12]-.036,0.),
                      np.where(contact,landing_error(ff)+18.*np.maximum(ff[:,8]-.043,0.),18.+31.*ff[:,0]))
        for i in np.flatnonzero(legal):
            ncmd=min(u.shape[1],int(np.ceil((y[i,1]+.12)/a.control_dt)))
            result.append(dict(tape=u[i,:ncmd].tolist(),outcome=y[i].tolist(),feedback=ff[i].tolist(),
                               body=optical.plain(near[ids[i]]),score=float(loss[i]),interior=bool(interior[i])))
        log.append(dict(trials=batch,touched=int(contact.sum()),legal=int(legal.sum()),interior=int(interior.sum()),
                        smallest_radius_legal=None if not legal.any() else float(y[legal,12].min())))
        archive.extend(zip(loss.tolist(),ids.copy(),pars.copy(),strict=True)); elite=sorted(archive,key=lambda a:a[0])[:16]
        chosen=rng.integers(len(elite),size=batch); ids=np.asarray([elite[k][1] for k in chosen]);
        pars=np.asarray([elite[k][2] for k in chosen])+rng.normal(size=pars.shape)*scales*(.50-.08*turn)
        pars=np.clip(pars,-2.65*scales,2.65*scales)
        ids[:len(elite)]=[e[1] for e in elite];pars[:len(elite)]=[e[2] for e in elite]
    return sorted(result,key=lambda w:(not w['interior'],w['score'])),log


def received(a,clear,c,old,leg,prefix,selected,known):
    before=float(a.data.time);guarded(a,clear);notes=[]
    try:
        with physical_contacts(a,notes):
            ans=long.physical_finish(a,old,c,leg,prefix,selected)
        clear_result=a.clearance_result()
        if clear_result['samples']!=round((a.data.time-before)/a.model.opt.timestep):
            raise ValueError('An actual suffix omitted a physical CAD state')
        if known:
            if any(w['target']!=list(school.STD(c,j,'random',None)[0]) for j,w in enumerate(ans['placements']['requested'])):
                raise ValueError('Changed official target law')
        ans.update(clearance=clear_result,native_contacts=notes)
        return ans
    finally:
        a.sample_clearance(False)


def run():
    p=_read(PLAN); row,earned,ready,old=school.teacher(); c=row['card']; leg=row['case']['leg']; offer=old['placements']['delivered'][leg]
    if p['midline_protocol_sha256']!=_hash(school.PLAN):raise ValueError('Changed forecourt family boundary')
    d=dict(plan_sha256=_hash(PLAN),row=row['case'],routes=[],status='searching')
    with TemporaryDirectory(prefix='tt55-midline-shadow-') as dirpath,mpc.centered(ready):
        a=mpc.arena(width=64,height=48);cad=mpc.arena(width=64,height=48,detail='cad');clear=CourtClearance(cad.model,articulated=True)
        try:
            src=Path(native.__file__).with_name('tt29_rally_rollout.cpp'); txt=src.read_text(); marker='y < -1.82'
            if txt.count(marker)!=1:raise ValueError('Shadow cutoff source changed')
            tmp=Path(dirpath)/'margin.cpp';tmp.write_text(txt.replace(marker,f'y < {p["alternate_shadow_limit"]}'))
            engine=Rollout(a,1);engine.trial=long.build(tmp,Path(dirpath)); d['shadow_sha256']=_hash(tmp)
            fork,late,table=long.indexed(old['events'],offer,a.control_dt)
            reference=[e for e in old['events'] if e['time']<=late*a.control_dt+1e-8]
            native.replay_prefix(a,old,c,fork);q0=a.data.ctrl[a._arm_ids[1][2]][:5].copy();work=long.Workspace(a,np.asarray(ready))
            route_seeds=[np.asarray(x['q']) for x in row['supported']]
            path=grids(work,table,p['depths'])
            geometry,notes=candidate_geometry(work,path,q0,(late-fork)*a.control_dt,route_seeds)
            routes=route_selection(geometry,p['maximum_routes']); d.update(geometry=notes,routed_goals=optical.plain(geometry),fork=fork,cut=late)
            optical.save(OUT,d)
            print('shape',len(geometry),[(z['p'][1],z['face'],z['radial_m']) for z in routes],flush=True)
            for j,z in enumerate(routes):
                actual,same,_=long.physically_prepare(a,old,c,fork,late,z,reference,brisk_flip=True)
                r=dict(index=j,geometry=optical.plain(z),same_early_ball=same,model=[],physically_tested=[]); d['routes'].append(r)
                if same:
                    saved=State(a)
                    try:
                        a.clearance_margin=.0006
                        candidates,summary=physical_model(a,engine,z,geometry,table['time'],p['native_batch'],p['rounds'],[550093,c['seed'],j])
                    finally:
                        a.clearance_margin=0.
                    r['model']=summary
                    print('model',j,[(t['legal'],t['interior'],t['smallest_radius_legal']) for t in summary],flush=True)
                    for k in candidates[:3]:
                        if not k['interior']:break
                        saved.restore(a)
                        live=received(a,clear,c,old,leg,actual,k,True)
                        r['physically_tested'].append(live)
                        if live['answered'] and not live['clearance']['penetration_samples']:
                            live['full_cad']=mid.full_verify(ready,c,live,clear)
                            if not live['full_cad']['penetration_samples']:
                                print('physical',j,live['full_cad'],flush=True)
                                d.update(status='physical interior witness',chosen_route=j)
                                break
                optical.save(OUT,d)
                if 'chosen_route' in d:break
            d['status']=d['status'] if 'chosen_route' in d else 'No certified interior strike from budgeted reachable geometry'
            d['complete']=True;optical.save(OUT,d)
        finally:
            a.close();cad.close()


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('mode',choices=('plan','run','audit'))
    args=a.parse_args()
    if args.mode=='plan':freeze()
    elif args.mode=='audit':audit_finite()
    else:run()
