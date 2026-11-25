"""Late falling low-ball probe on the unchanged nominal SO101 and matched real offers."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from . import tt38_joint_mpc as mpc
from . import tt40_long_low as long
from . import tt55_midline_margin as prior
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import Rollout, _aero, desired_shot
from .tt39_opportunity_map import projected_height
from .tt44_paired_credit import State
from .tt54_midline import full_verify
from .tt54_training_speedups import enabled
from .tt37_opponent_audit import CourtClearance

ROOT=Path('artifacts/tt64_deep_receive')
PLAN=ROOT/'protocol.json'
OUT=ROOT/'deep.json.gz'
OLD=Path('artifacts/tt54_midline/posture.json.gz')
SEASON=Path('artifacts/tt46_season/learning.json.gz')
DEPTHS=np.array([-1.54,-1.60,-1.66,-1.72,-1.78,-1.84,-1.90,-1.96,-2.02,-2.08,-2.14])
SURFACE_RADIUS=.052
REAL_RADIUS=.052
GAP=.024
PHASES=2
BATCH=96
MAX_ROUTES=4
AIM=mpc.AIM.copy()


def docket():
    d=_read(SEASON)
    seen=set(); sources={str(SEASON):_hash(SEASON),str(OLD):_hash(OLD)}
    chosen=[]
    anchor=_read(OLD)['rows'][0]['case']
    chosen.append(dict(id=anchor['id'],index=anchor['index'],leg=anchor['leg'],height=anchor['h'],x=anchor['lat'],source=anchor['source']))
    for height0,height1 in ((.978,.999),(.95,.978)):
        for j,(c,row) in enumerate(zip(d['cards'],d['rows'])):
            if c['id'] in seen or c['profile']!='reach' or not row['placements']['delivered']:
                continue
            o=row['placements']['delivered'][-1]
            if not o['leg'] or o['near_landed'] or o['near_contact'] or not o['receive_line'] or not o['near_bounce']:
                continue
            h=projected_height(o['receive_line']); x=o['receive_line']['xyz'][0]
            if abs(x)>.075 or not height0<=h<height1:
                continue
            if any(w['draw']!='random' for w in row['placements']['requested']):continue
            chosen.append(dict(id=c['id'],index=j,leg=o['leg'],height=h,x=x,source=str(SEASON)))
            seen.add(c['id']);break
    if len(chosen)!=3:raise ValueError('Three natural untouched central low offers required')
    sources[anchor['source']]=_hash(Path(anchor['source']))
    if sources[anchor['source']]!=anchor['source_sha256']:raise ValueError('Prior reference deteriorated')
    _write(PLAN,dict(schema='tt64-deep-underpass-1',cases=chosen,sources=sources,depths_m=DEPTHS.tolist(),
             limits=dict(interior_radius_m=REAL_RADIUS,paths=MAX_ROUTES,batches=PHASES,commands_per_batch=BATCH),
             role='Mechanism-only training oracle on exact archived actual near bounces. Changed control begins after the preceding own blade separation and remains in clearance. Ceiling fit is centered impact area, face sign, and joint-velocity budget down behind the endline, not a late rim fit. Two untouched ordinary season misses supplement the previously studied 4214. All real grading uses the original source fixture, opponent draws, timing, motor envelope, CAD and referee. No live policy accesses chronology or trajectory truth.'))
    print('frozen cases',chosen,flush=True)


def deep_track(a,separation):
    """Only the isolated analysis supports a lower/deeper numerical termination."""
    f=a.partner.flight; m,d=f.model,f.data
    mujoco.mj_resetData(m,d)
    d.qpos[:]=np.r_[separation['ball'],1.,0.,0.,0.]
    d.qvel[:]=np.r_[separation['velocity'],separation['spin']]
    mujoco.mj_forward(m,d)
    data=[]
    for _ in range(round(.85/m.opt.timestep)):
        spin=np.empty(3); mujoco.mju_rotVecQuat(spin,d.qvel[3:],d.qpos[3:])
        d.xfrc_applied[1]=_aero(d.qvel[:3],spin,f.drag,f.lift)
        mujoco.mj_step(m,d)
        p,v=d.qpos[:3],d.qvel[:3]
        if p[1]<-2.18 or p[2]<.60 or v[1]>=-.25 or any(c.efc_address>=0 for c in d.contact):break
        if p[1]<-1.52:
            data.append(np.r_[d.time,p,v,spin])
    a=np.asarray(data).reshape(-1,10)
    if not len(a):return a
    return a[np.unique([int(np.argmin(abs(a[:,2]-s))) for s in DEPTHS if a[:,2].min()-.015<s<a[:,2].max()+.015])]


def nrm(x):return x/max(1.e-9,np.linalg.norm(x))


def geometry_fit(ik,p,want,f,seed):
    lo,hi=ik.low+1.e-8,ik.high-1.e-8
    def terms(q):
        c,n,j,jN=ik.pose(q,jacobian=True); n=n*f; jN=jN*f
        d=p-c; ax=d@n; dax=d@jN-n@j
        r=d-ax*n; r_j=-j-n[:,None]*dax-ax*jN
        radius=np.linalg.norm(r)
        return n,j,jN,ax,dax,radius,(r@r_j)/max(1.e-9,radius)
    def ev(q):
        n,j,jN,ax,Dax,r,Dr=terms(q)
        return np.r_[2.3*(ax-GAP),2.0*max(0.,r-SURFACE_RADIUS+.006),.093*(n-want),.003*(q-seed)]
    def jac(q):
        n,j,jN,ax,Dax,r,Dr=terms(q)
        return np.r_[2.3*Dax[None,:],2.*(r>SURFACE_RADIUS-.006)*Dr[None,:],.093*jN,.003*np.eye(5)]
    ans=least_squares(ev,np.clip(seed,lo,hi),jac=jac,bounds=(lo,hi),max_nfev=30,ftol=2.e-4)
    n,j,jN,ax,Dax,r,Dr=terms(ans.x)
    return ans.x,n,j,jN,float(r),float(ax)


def speed_family(q,n,j,jN,lo,hi,v,ideal):
    """Choose a normal impulse from the joint velocity box, not a wrist-only poke."""
    prep=.093; through=.051
    # Lower and upper rate both leave room for the preceding backswing and
    # the following 51ms braking interval inside the original hard limits.
    lower=np.maximum(-4.2,np.maximum((q-hi)/prep,(lo-q)/through))
    upper=np.minimum(4.2,np.minimum((q-lo)/prep,(hi-q)/through))
    proj=n@j; max_dir=np.where(proj>=0.,upper,lower)
    max_speed=max(0.,float(proj@max_dir))
    target=max(.13,float(ideal)); drive=max_dir*min(1.,target/max(.01,max_speed))
    # An alternative uses the minimum angular disturbance for a forward/upward
    # center velocity. Saturate against the same directional travel bounds.
    ta=np.array([0.,0.,1.])-n*n[2]
    sys=np.r_[j,.041*jN,.017*np.eye(5)]
    v_want=target*n+(.10+.12*max(0.,-v[2]))*ta
    smooth=np.clip(np.linalg.lstsq(sys,np.r_[v_want,np.zeros(8)],rcond=None)[0],lower,upper)
    return drive,smooth,max_speed


def route(guard,home,approach,lo,hi):
    if not guard.clear(approach):return None
    middle=.53*home+.47*approach
    for off in ((0,0,0,0,0),(0,.26,.38,-.14,0),(0,.44,.21,.09,0),(.28,.22,.35,-.21,0),(-.28,.22,.35,-.21,0)):
        w=np.clip(middle+off,lo,hi)
        if guard.route(home,w,steps=14) and guard.route(w,approach,steps=14):return w
    return None


def scan(w,track,home,t0,max_time):
    ik=w.ik; poses=[]; bounds=[]; all_old=_read(prior.OUT)['routed_goals']; old=[np.asarray(r['q']) for r in all_old if r['p'][1]<-1.55]
    for eta,*r in track:
        p,v=np.array(r[:3]),np.array(r[3:6]); best={}; fitted=0; feasible=0
        for pace in (3.7,4.65,5.5,6.3):
            for landing in (.43,.86):
                desired=desired_shot(p,v,1,[0.,landing],pace)
                for z in (-.11,.0,.12):
                    want=nrm(desired+[0.,0.,z]); target_vy=float(pace)
                    for face in (-1,1):
                        _,idx=w.index.query(np.r_[p-GAP*want,face*want]/long.SCALE,k=20)
                        q0=w.poses[idx[0]]; q1=w.poses[idx[np.argmax(np.linalg.norm(w.poses[idx]-q0,axis=1))]]
                        for seed in (home,q0,q1,*old[:2]):
                            q,n,j,N,rad,ax=geometry_fit(ik,p,want,face,seed)
                            required=(target_vy-v[1])/(1.80*max(n[1],.08))+v@n
                            dc,ds,mx=speed_family(q,n,j,N,ik.low,ik.high,v,required)
                            grade=2.2*max(0.,rad-SURFACE_RADIUS)+abs(ax-GAP)+.064*np.linalg.norm(n-want)+.055*max(0.,required-mx)
                            item=dict(depth=float(p[1]),z=float(p[2]),vy=float(v[1]),vz=float(v[2]),face=face,radial_m=rad,
                                      axial_m=ax, normal=n.tolist(),joint=q.tolist(),wanted=want.tolist(),
                                      needed_normal_mps=float(required),ceiling_normal_mps=mx,grade=float(grade))
                            if str(face) not in best or grade<best[str(face)]['grade']:best[str(face)]=item
                            if rad>SURFACE_RADIUS or abs(ax-GAP)>.011 or n[1]<.49 or n[2]<.12 or n@want<.84:
                                continue
                            fitted+=1
                            if (not w.guard.clear(q) or np.max(abs(home-q))>max_time*3.45 or required>mx+.23):continue
                            if any(k['face']==face and abs(k['p'][1]-p[1])<.022 and np.linalg.norm(k['q']-q)<.19 for k in poses):continue
                            for mode,dq in (('normal-ceiling',dc),('smooth-lifting',ds)):
                                if n@j@dq < max(.08,required-.28):continue
                                approach=q-.093*dq
                                waypoint=route(w.guard,home,approach,ik.low,ik.high)
                                if waypoint is None:continue
                                feasible+=1
                                y_speed=v[1]+1.80*((j@dq-v)@n)*n[1]
                                z_speed=v[2]+1.80*((j@dq-v)@n)*n[2]
                                t_net=abs(p[1])/max(.4,.90*y_speed)
                                net=float(p[2]+t_net*z_speed-4.905*t_net*t_net)
                                t_land=max(0.,(z_speed+np.sqrt(max(0.,z_speed*z_speed+19.62*(p[2]-.779))))/9.81)
                                landing_y=p[1]+.90*y_speed*t_land
                                cost=5.*max(0.,.993-net)+1.3*abs(landing_y-.72)+.6*rad+.002*abs(p[1]+1.84)
                                poses.append(dict(time=t0+float(eta),p=p,v=v,face=face,q=q,dq=dq,approach=approach,
                                    waypoint=waypoint,normal=n,radius=rad,mode=mode,proxy_net=net,
                                    proxy_land=float(landing_y),proxy_speed=float(y_speed),cost=float(cost)))
        bounds.append(dict(depth=float(p[1]),height=float(p[2]),descending=float(v[2]),best=best,
                           center_direction_fits=fitted,preload_routes=feasible))
    return poses,bounds


def select(poses):
    order=sorted(poses,key=lambda row:row['cost']); keep=[]
    # Span deep, medium and reverse if the reach map supports them.
    for upper,lower in ((-1.79,-2.20),(-1.52,-1.79)):
        for f in (-1,1):
            q=next((v for v in order if lower<=v['p'][1]<upper and v['face']==f),None)
            if q is not None:keep.append(q)
    for row in order:
        if len(keep)>=MAX_ROUTES:break
        if all(np.linalg.norm(row['q']-q['q'])>.20 or abs(row['p'][1]-q['p'][1])>.10 for q in keep):keep.append(row)
    return keep[:MAX_ROUTES]


def landing_proxy(f):
    vx,vy,vz=f[:,2],f[:,3],f[:,4]; px,py,pz=f[:,5],f[:,6],f[:,7]
    t=np.maximum(0.,(vz+np.sqrt(np.maximum(0.,vz*vz+19.62*(pz-.779))))/9.81)
    net=-py/np.maximum(.40,.89*vy)
    h=pz+vz*net-4.905*net*net
    return (5.*np.maximum(0.,.97-h)+2.3*abs(py+.89*vy*t-.72)+2.*abs(px+vx*t)+3.*np.maximum(0.,2.-vy))


def overrun(a,engine,pose,pool,seed):
    near=sorted([q for q in pool if q['face']==pose['face'] and np.linalg.norm(q['q']-pose['q'])<.58],
                key=lambda q:q['cost']+.13*np.linalg.norm(q['q']-pose['q']))[:12]
    near=[pose,*[q for q in near if q is not pose]]
    now=float(a.data.time); dt=a.control_dt; budget=max(q['time']-now for q in near)+.10
    n=max(77,int((budget+1.15)/dt))
    q=np.array([g['q'] for g in near]); dq=np.array([g['dq'] for g in near]); eta=np.array([g['time']-now for g in near])
    t=(1+np.arange(n))[None,:]*dt; rng=np.random.default_rng(seed)
    size=np.array([.034,.030,.052,.049,.083,.65,.70,.93,.86,1.1,.033,.024,.040])
    rows=[]; archive=[]; hits=[]; n_hits=[]
    z=(2*qmc.Sobol(13,scramble=True,seed=int(rng.integers(2**30))).random_base2(7)[:BATCH]-1)*size
    ix=np.arange(BATCH)%len(q);z[:len(q)]=0.
    for j in range(PHASES):
        # Acceleration from beneath or behind, then a short normal drive and brake.
        wave=t-eta[ix,None]+(.038+z[:,10,None])
        stride=np.clip(wave,-.093,.070)
        g=q[ix,None,:]+z[:,None,:5]+stride[:,:,None]*(dq[ix,None,:]+z[:,None,5:10])
        g[:,:,3:5]+=np.clip(wave/.080,0.,1.)[:,:,None]*z[:,None,11:13]
        g=np.concatenate((g,np.full((*g.shape[:2],1),a.data.ctrl[engine.axes[-1]])),axis=2)
        fb=np.zeros((len(z),9)); appr=np.full((len(z),19),np.nan)
        u,y=engine.trajectories(g,AIM,budget,feedback=fb,approaches=appr)
        sep=fb[:,1]>.5; legal=np.isfinite(y[:,0]); inside=sep&(fb[:,8]<=REAL_RADIUS)
        quality=np.where(legal,mpc.cost(y,fb)[0]-14.,np.where(sep,landing_proxy(fb),25.+19.*fb[:,0]))
        quality+=np.where(sep,45.*np.maximum(0.,fb[:,8]-REAL_RADIUS),0.)
        rows.append(dict(trials=len(z),physical_candidate_contact=int(sep.sum()),interior_contact=int(inside.sum()),
                         model_legal=int(legal.sum()),legal_interior=int((legal&inside).sum()),
                         best_interior_forward_mps=None if not inside.any() else float(max(fb[inside,3]))))
        for k in np.flatnonzero(inside)[np.argsort(quality[inside])[:8]]:
            record=dict(rank=float(quality[k]),candidate_radius_m=float(fb[k,8]),candidate_departure=fb[k,2:8].tolist(),
                        model_legal=bool(legal[k]),model_contact_step=int(np.ceil(appr[k,0]/dt)),rate=u[k].copy(),
                        variant=long.plain(near[ix[k]]),parameters=z[k].tolist())
            n_hits.append(record)
            if legal[k]:hits.append(record)
        archive+=list(zip(quality,ix.copy(),z.copy(),strict=True))
        weak=sorted(archive,key=lambda a:a[0])[:17]
        if j+1<PHASES:
            sel=rng.integers(len(weak),size=BATCH); ix=np.array([weak[k][1] for k in sel]); z=np.array([weak[k][2] for k in sel])
            z+=rng.normal(size=z.shape)*size*.54; z=np.clip(z,-1.80*size,1.80*size)
            for k,z0 in enumerate(weak):ix[k]=z0[1];z[k]=z0[2]
    # At most two modeled legal interior strokes and one finite interior
    # negative. The latter guards against shadow false-negative counts.
    accepted=[]
    for c in sorted(hits,key=lambda c:c['rank'])+sorted(n_hits,key=lambda c:c['rank']):
        t=c['rate'][:c['model_contact_step']+8]
        if len(accepted)>=3 or sum(not v['model_legal'] for v in accepted)>=1 and not c['model_legal']:continue
        if any(np.max(abs(v['tape'][:9]-t[:9]))<.035 for v in accepted):continue
        c=c.copy();c.pop('rate');c['tape']=t
        accepted.append(c)
    return accepted,rows


def run():
    plan=_read(PLAN)
    if any(_hash(Path(k))!=h for k,h in plan['sources'].items()):raise ValueError('Changed matched archive')
    result=dict(protocol_sha256=_hash(PLAN),rows=[],complete=False)
    with TemporaryDirectory(prefix='tt64-deep-shadow-') as tmp,enabled():
        src=Path(long.__file__).with_name('tt29_rally_rollout.cpp'); body=src.read_text()
        if body.count('y < -1.82')!=1 or body.count('z < .70')!=1:raise ValueError('Changed native shadow depth cutoff')
        cpp=Path(tmp)/'deep.cpp'; cpp.write_text(body.replace('y < -1.82','y < -2.18').replace('z < .70','z < .584'))
        trial=long.build(cpp,Path(tmp)); result['sink_shadow_sha256']=_hash(cpp)
        for spec in plan['cases']:
            started=process_time(); d=_read(Path(spec['source'])); card,row=d['cards'][spec['index']],d['rows'][spec['index']]; leg=spec['leg']
            offer=row['placements']['delivered'][leg]
            with mpc.centered(d['ready']):
                a=mpc.arena(width=64,height=48); cad=mpc.arena(width=64,height=48,detail='cad'); guard=CourtClearance(cad.model,articulated=True)
                setting=dict(case=spec,card=card,ready=d['ready'],routes=[])
                try:
                    fork,cut,table=long.indexed(row['events'],offer,a.control_dt)
                    received=[e for e in row['events'] if e['time']<=cut*a.control_dt+1.e-8]
                    long.replay_prefix(a,row,card,fork)
                    home=a.data.ctrl[a._arm_ids[1][2]][:5].copy()
                    w=long.Workspace(a,d['ready']); track=deep_track(a,table)
                    poses,ends=scan(w,track,home,table['time'],(cut-fork)*a.control_dt)
                    candidates=select(poses); engine=Rollout(a,1);engine.trial=trial
                    setting.update(fork=fork,receiving=cut,table_separation=table,scan=ends,
                                   scanned_depths=len(track),supported_geometries=len(poses),used=long.plain(dict(track=track)))
                    print('geometry',spec['id'],len(track),len(poses),[(v['p'][1],v['p'][2],v['face'],v['proxy_net'],v['proxy_land']) for v in candidates],flush=True)
                    for p in candidates:
                        prefix,matched,_=long.physically_prepare(a,row,card,fork,cut,p,received,brisk_flip=True)
                        note=dict(pose=long.plain(p),same_prefix=matched,physical=[])
                        setting['routes'].append(note)
                        if not matched:continue
                        saved=State(a)
                        plans,screen=overrun(a,engine,p,poses,[640126,card['seed'],len(setting['routes'])]);note['screen']=screen
                        print('screen',spec['id'],screen,flush=True)
                        for m in plans:
                            saved.restore(a); option=dict(m,tape=m['tape'].tolist())
                            actual=prior.received(a,guard,card,row,leg,prefix,option,True)
                            touched=[c for c in actual['native_contacts'] if c['side']==1]
                            best=touched[0] if touched else None
                            if actual['answered'] and actual['clearance']['penetration_samples']==0 and best is not None and best['radial_m']<=REAL_RADIUS and abs(best['face_alignment'])>=.95:
                                actual['full_cad']=full_verify(d['ready'],card,actual,guard)
                                if not actual['full_cad']['penetration_samples']:
                                    setting['witness']=True
                            note['physical'].append(actual)
                            print('physical',spec['id'],actual['legal'],actual['answered'],best,flush=True)
                            if setting.get('witness'):break
                        if setting.get('witness'):break
                    setting['cpu_s']=process_time()-started
                finally:
                    cad.close();a.close()
            result['rows'].append(setting)
            result['complete']=len(result['rows'])==len(plan['cases'])
            _write(OUT,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=('plan','run')); a=p.parse_args()
    ROOT.mkdir(exist_ok=True,parents=True)
    docket() if a.mode=='plan' else run()
