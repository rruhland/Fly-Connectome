"""Physical long-preparation feasibility test for three previously unresolved low balls."""

import argparse
import ctypes
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import process_time

import numpy as np
import mujoco
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import Rollout, _aero, _native_trials
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_bodycoach_windows import Reach
from .tt37_motor_clinic import landing_error
from .tt37_opponent import arena
from .tt38_motorcoach import AIM, SIZES, answerable, schedules
from .tt39_terminal_servo import desired_pose
from .tt40_low_feasibility import Faces, SOURCE, ZERO, replay_prefix, roster

OUT = Path('artifacts/tt40')
ORDER = (14843, 4322, 8302)
DEPTHS = (-1.08,-1.18,-1.28,-1.36,-1.44,-1.53,-1.62,-1.71,-1.78,-1.85,-1.92)
SCALE = np.r_[[.14,.116,.105], [.42,.42,.42]]
PRIVATE_Y_LIMIT = -1.97


def build(source, folder):
    """Compile an independent shared object without rewriting the default source."""
    package=Path(mujoco.__file__).parent
    lib=next(package.glob('libmujoco.so.*'))
    target=folder/(source.stem+'.so')
    subprocess.run(['c++','-std=c++17','-O3','-ffp-contract=off','-shared','-fPIC','-fopenmp','-I',
                    str(package/'include'),str(source),str(lib),f'-Wl,-rpath,{package}','-o',str(target)],check=True)
    fn=ctypes.CDLL(str(target)).trial_strokes_v5
    fn.argtypes=_native_trials()[0].argtypes
    fn.restype=None
    return fn


def same_trace(a,b):
    """Keep contact sequence and placement discrete values exact; allow solver roundoff."""
    if isinstance(a,np.generic):
        a=a.item()
    if isinstance(b,np.generic):
        b=b.item()
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(same_trace(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return isinstance(b,(list,tuple)) and len(a)==len(b) and all(same_trace(x,y) for x,y in zip(a,b,strict=True))
    if isinstance(a,float) and isinstance(b,(float,int)) and not isinstance(b,bool):
        return abs(a-b)<=1e-8
    return type(a)==type(b) and a==b


def event_difference(live,archive):
    if len(live)!=len(archive):
        return dict(events_live=len(live),events_archived=len(archive))
    return next((dict(index=k, live=a, archived=b) for k,(a,b) in enumerate(zip(live,archive,strict=True))
                 if not same_trace(a,b)),None)


def indexed(events, offered, dt):
    shot = offered['far_contact']['time']
    prev = [e for e in events if e['kind']=='separation' and e.get('collider')=='learner_racket' and e['time']<shot]
    stop = [e for e in events if e['kind']=='separation' and e.get('collider') in ('table','table_right')
            and offered['near_bounce']['time']<e['time']<offered['near_bounce']['time']+.06]
    if not prev or not stop:
        raise ValueError('Expected one preceding near and one target table separation')
    tick = lambda t: int(np.ceil((t+1e-7)/dt))+1
    return tick(prev[-1]['time'])+3, tick(stop[0]['time']), stop[0]


def ease(t):
    t=np.clip(t,0.,1.)
    return t*t*(3.-2.*t)


def preparation(q0, waypoint, final, n, grip, brisk=False):
    """Three position knots with a smooth stop or a faster roll-first route."""
    m=max(1,round(.47*n))
    ss=(np.arange(m)+1)/m
    tt=(np.arange(n-m)+1)/(n-m)
    s=.85*ss+.15*ease(ss) if brisk else ease(ss)
    t=.85*tt+.15*ease(tt) if brisk else ease(tt)
    a=(1-s[:,None])*q0+s[:,None]*waypoint
    b=(1-t[:,None])*waypoint+t[:,None]*final
    return np.c_[np.r_[a,b],np.full(n,grip)]


def defensive_flight(flight, separation):
    """Isolated post-bounce ball, keeping sub-table-height samples until z=.71 m."""
    m,d=flight.model,flight.data
    mujoco.mj_resetData(m,d)
    d.qpos[:]=np.r_[separation['ball'],1.,0.,0.,0.]
    d.qvel[:]=np.r_[separation['velocity'],separation['spin']]
    mujoco.mj_forward(m,d)
    matched=[]
    while d.time<.75:
        w=np.empty(3)
        mujoco.mju_rotVecQuat(w,d.qvel[3:],d.qpos[3:])
        d.xfrc_applied[1]=_aero(d.qvel[:3],w,flight.drag,flight.lift)
        mujoco.mj_step(m,d)
        p,v=d.qpos[:3],d.qvel[:3]
        touch=any(c.efc_address>=0 and (c.geom[0] in flight.surface or c.geom[1] in flight.surface)
                  for c in d.contact)
        if touch or p[1]<-1.98 or p[2]<.71 or v[1]>-.20:
            break
        matched.append(np.r_[d.time,p,v,w])
    return np.asarray(matched).reshape(-1,10)


def plain(value):
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in value.items()}


class Workspace:
    def __init__(self,a,start):
        self.a=a
        self.ik=Faces(a)
        self.guard=Reach(a,start,start)
        self.poses=self.guard.q
        self.index=cKDTree(np.array([np.r_[*self.ik.pose(q)] for q in self.poses])/SCALE)

    def windows(self, table, defensive=False):
        flight=self.a.partner.flight
        x=(defensive_flight(flight,table) if defensive else
           flight.path(table['ball'],table['velocity'],table['spin'],1,bounced=True,horizon=.80))
        if not len(x):
            return []
        depths=(-1.72,-1.76,-1.80,-1.84,-1.88,-1.92) if defensive else DEPTHS
        return [x[int(np.argmin(abs(x[:,2]-y)))] for y in depths
                if x[:,2].min()-.014<y<x[:,2].max()+.014]

    def support(self, track, q0, budget):
        good=[]
        failed=[]
        for row in track:
            time, p, v = row[0], row[1:4], row[4:7]
            if not .71<p[2]<1.25:
                continue
            for face in (-1,1):
                candidates=[]
                for pace in (3.6,4.3,4.95):
                    aim=desired_pose(p,v,pace)
                    for tilt in (-.09,.08,.22):
                        n=aim[3:6]+np.array([0.,0.,tilt])
                        n/=np.linalg.norm(n)
                        goal=p-.030*n
                        _,ii=self.index.query(np.r_[goal,face*n]/SCALE,k=14)
                        starts=[q0,self.poses[ii[0]]]
                        j=int(np.argmax(np.max(abs(self.poses[ii]-starts[-1]),axis=1)))
                        if np.linalg.norm(self.poses[ii[j]]-starts[-1])>.47:
                            starts.append(self.poses[ii[j]])
                        for seed in starts:
                            q,(center,axis,jp,jn)=self.ik.fit(goal,face*n,seed)
                            radius=np.linalg.norm(p-center-.025*face*axis)
                            aligned=n@(face*axis)
                            distance=np.linalg.norm(goal-center)
                            fit=distance+.08*(1.-aligned)
                            if distance>.060 or aligned<.88 or radius>.079:
                                failed.append((float(p[1]),face,float(fit)))
                                continue
                            if not self.guard.clear(q):
                                continue
                            if np.max(abs(q-q0))>budget*3.5:
                                continue
                            momentum=max(.50,aim[6])*n+np.array([-.34*v[0],.05,.55])
                            dq=np.linalg.lstsq(np.r_[jp,.05*jn,.021*np.eye(5)],
                                                np.r_[momentum,np.zeros(8)],rcond=None)[0].clip(-6.,6.)
                            approach=q-.056*dq
                            if not self.guard.clear(approach):
                                approach=q.copy()
                            waypoint=None
                            for mid in (.50*(q0+approach),
                                        .50*(q0+approach)+[0,.26,.33,-.12,0],
                                        .50*(q0+approach)+[0,.45,.17,.10,0],
                                        .50*(q0+approach)+[.17,.25,.31,.15,0]):
                                mid=np.clip(mid,self.ik.low,self.ik.high)
                                if self.guard.route(q0,mid,steps=10) and self.guard.route(mid,approach,steps=10):
                                    waypoint=mid
                                    break
                            if waypoint is None:
                                continue
                            score=fit+.35*radius+.024*np.linalg.norm(q-q0)+.012*abs(pace-4.3)
                            candidates.append(dict(time=time,p=p,v=v,face=face,q=q,dq=dq,n=n,
                                                   approach=approach,waypoint=waypoint,pace=pace,quality=score))
                ranked=sorted(candidates,key=lambda d:d['quality'])
                for k in ranked:
                    if any(r['face']==face and abs(r['p'][1]-p[1])<.015
                           and np.linalg.norm(r['q']-k['q'])<.22 for r in good):
                        continue
                    if sum(r['face']==face and abs(r['p'][1]-p[1])<.015 for r in good)>=3:
                        break
                    good.append(k)
        return good,failed


def select(bases,max_ports):
    # Group by signed face and by depth, exposing the deeper area even when shallow fits exist.
    buckets=[]
    for bounds in ((-1.55,0),(-1.78,-1.55),(-2.,-1.78)):
        for f in (-1,1):
            b=sorted([a for a in bases if a['face']==f and bounds[0]<=a['p'][1]<bounds[1]],
                     key=lambda a:a['quality'])
            if b:
                buckets.append(b)
    # Try the best supported geometry first, then distinct alternative sides/depths.
    buckets.sort(key=lambda x:x[0]['quality'])
    potent=[b[0] for b in buckets]
    tail=sorted([a for b in buckets for a in b[1:]],key=lambda a:a['quality'])
    potent.extend(a for a in tail if all(np.linalg.norm(a['q']-l['q'])>.26 for l in potent))
    return potent[:max_ports]


def physically_prepare(a,old,card,fork,late,target,expect,brisk_flip=False):
    stamp=replay_prefix(a,old,card,fork)
    axes=a._arm_ids[1][2]
    stride=a._joint_increments[axes]*(a.control_dt/a.model.opt.timestep)
    qstart=a.data.ctrl[axes].copy()
    brisk=brisk_flip and target['face']<0
    proposed=preparation(qstart[:5],target['waypoint'],target['approach'],late-fork,qstart[-1],brisk)
    actions=list(old['actions'][:fork])
    for g in proposed:
        bound=.80 if brisk else .71
        z=np.clip((g-a.data.ctrl[axes])/stride,-bound,bound)
        _,_,end,cap,_=a.exchange(z,FAR_COMMAND)
        actions.append(z.tolist())
        if end or cap:
            break
    nochange=len(actions)==late and same_trace(a.events,expect) and not (a.terminated or a.truncated)
    return actions,nochange,stamp


def late_search(a,engine,prepared,bases,batch,rounds,seed):
    near=[b for b in bases if np.linalg.norm(b['q']-prepared['q'])<.90 and b['face']==prepared['face']]
    if not near:
        near=[prepared]
    near.sort(key=lambda b:np.linalg.norm(b['q']-a.joints[:5])+.45*abs(b['p'][1]-prepared['p'][1]))
    near=near[:18]
    # Exact time origin uses event table separation plus the sample-deduced current offset.
    et=np.asarray([b['eta'] for b in near])
    good_time=np.array([-0.016 < t < .55 for t in et])
    kept=np.flatnonzero(good_time)
    if not len(kept):
        return [],[]
    near=[near[k] for k in kept]
    et=et[kept]
    q=np.asarray([b['q'] for b in near]); w=np.asarray([b['dq'] for b in near])
    rng=np.random.default_rng(seed)
    raw=qmc.Sobol(len(SIZES),scramble=True,seed=int(rng.integers(1,2**31))).random_base2(
        int(np.ceil(np.log2(batch))))[:batch]
    z=(2*raw-1)*SIZES*.65
    keys=np.arange(batch)%len(near)
    z[:len(near)]=0.
    cap=float(np.max(et)+.140)
    length=round((cap+.92)/a.control_dt)
    archive=[]; found=[]; records=[]
    for idx in range(rounds):
        g=schedules(q[keys],w[keys],et[keys],z,a.control_dt,length)
        f=np.zeros((batch,9))
        rate,y=engine.trajectories(g,AIM,cap,feedback=f)
        legal=np.isfinite(y[:,0])
        grade=np.where(legal,answerable(y)-12.,np.where(f[:,1]>.5,landing_error(f)+2*f[:,8],18.+36*f[:,0]))
        for k in np.flatnonzero(legal):
            n=min(rate.shape[1],max(1,int(np.ceil((y[k,1]+.12)/a.control_dt))))
            found.append(dict(tape=rate[k,:n].tolist(),cost=float(answerable(y[k])),outcome=y[k].tolist(),
                              feedback=f[k].tolist(),variant=plain(near[keys[k]]),parameters=z[k].tolist()))
        hit=f[:,1]>.5
        proposal=None
        if hit.any():
            k=np.argmin(grade)
            proposal=f[k,2:8].tolist()
        records.append(dict(round=idx,tested=batch,touched=int(hit.sum()),private_legal=int(legal.sum()),
                            closest=float(f[:,0].min()),best_contact=proposal))
        if any(b['cost']<4.1 for b in found):
            break
        archive.extend(zip(grade,keys.copy(),z.copy(),strict=True))
        top=sorted(archive,key=lambda x:x[0])[:16]
        select_p=rng.integers(len(top),size=batch)
        keys=np.array([top[i][1] for i in select_p])
        z=np.array([top[i][2] for i in select_p])+rng.normal(size=(batch,len(SIZES)))*SIZES*(.63 if idx<2 else .38)
        z=np.clip(z,-2.8*SIZES,2.8*SIZES)
        z[:len(top)]=np.array([x[2] for x in top]); keys[:len(top)]=[x[1] for x in top]
    return sorted(found,key=lambda z:z['cost']),records


def physical_finish(a,old,card,leg,prefix,chosen):
    actions=list(prefix)
    for i in range(150):
        u=chosen['tape'][i] if i<len(chosen['tape']) else ZERO.tolist()
        _,_,end,cap,_=a.exchange(u,FAR_COMMAND)
        actions.append(u)
        if end or cap or a.rival_returns>=leg+2:
            break
    placements=a.point_placements()
    received=placements['delivered'][leg]
    old_offer=old['placements']['delivered'][leg]
    if not same_trace(received['far_contact'],old_offer['far_contact']):
        raise ValueError('Long preparation changed the target far impact')
    if not same_trace(received['near_bounce'],old_offer['near_bounce']):
        raise ValueError('Long preparation changed the delivered bounce')
    requests = [r['target'] for r in placements['requested'][:leg+1]]
    old_requests = [r['target'] for r in old['placements']['requested'][:leg+1]]
    if requests!=old_requests:
        raise ValueError('Random placement protocol changed at the branch')
    return dict(actions=actions,events=a.events,placements=placements,score=audit(a.events,opening_side=1),
                shot=old['shot'],legal=bool(received['near_landed']),
                answered=bool(received['near_landed'] and len(placements['delivered'])>leg+1),
                motor_audit=a.motor_audit(),chosen={k:v for k,v in chosen.items() if k!='tape'})


def run(output,max_ports,batch,rounds,case=None,face_filter=None,continue_after=False,skip_ports=0,
        brisk_flip=False,defensive=False,private_clearance=0.):
    source=_read(SOURCE)
    if source.get('group')!='practice':
        raise ValueError('Source must be declared development')
    picks={x['id']:x for x in roster(source)}
    tests=[picks[j] for j in ORDER if case is None or j in case]
    receipt=dict(schema='tt40-long-low-1',source_sha256=_hash(SOURCE),role='privileged preparation feasibility only',
                 ordered_releases=[t['id'] for t in tests],ready=source['ready'],depth_grid=DEPTHS,
                 private_y_cutoff=PRIVATE_Y_LIMIT,native_rounds=rounds,batch=batch,
                 face_filter=face_filter,continue_after=continue_after,skip_ports=skip_ports,
                 brisk_flip=brisk_flip,defensive_grid=defensive,private_clearance_m=private_clearance,rows=[])
    with TemporaryDirectory(prefix='tt40-long-low-') as name, centered(source['ready']):
        shared=Path(__file__).with_name('tt29_rally_rollout.cpp')
        body=shared.read_text()
        token='y < -1.82'
        if body.count(token)!=1:
            raise ValueError('Check independent optional deep-horizon source')
        test_path=Path(name)/'long_low_native.cpp'
        test_path.write_text(body.replace(token,f'y < {PRIVATE_Y_LIMIT}'))
        receipt['stock_native_sha256']=_hash(shared)
        receipt['private_native_sha256']=_hash(test_path)
        special=build(test_path,Path(name)) if max_ports else None
        a=arena(protocol='random')
        engine=Rollout(a,1) if max_ports else None
        if engine is not None:
            engine.trial=special
        work=Workspace(a,np.asarray(source['ready']))
        try:
            for target in tests:
                done=False
                tm=process_time()
                c=source['cards'][target['index']]; old=source['rows'][target['index']]; leg=target['leg']
                offered=old['placements']['delivered'][leg]; dt=a.control_dt
                fork,late,table=indexed(old['events'],offered,dt)
                front=replay_prefix(a,old,c,fork)
                q=a.data.ctrl[a._arm_ids[1][2]][:5].copy()
                observed=[e for e in old['events'] if e['time']<=late*dt+1e-8]
                expect=observed
                terms=work.windows(table,defensive)
                poses,failed=work.support(terms,q,(late-fork)*dt)
                for p in poses:p['eta']=float(p['time']+table['time']-late*dt)
                picked=select([x for x in poses if face_filter is None or x['face']==face_filter],
                              max_ports+skip_ports)[skip_ports:]
                item=dict(target,card=c,fork_tick=fork,late_tick=late,preparation_s=(late-fork)*dt,
                          roster_bounds=[v[1:4].tolist() for v in terms],prefix_events_sha256=front,
                          geometry={str(v):sum(p['face']==v for p in poses) for v in (-1,1)},
                          all_clear_routed_possibilities=[plain(p) for p in poses],
                          rejected_pose_scores=len(failed),trials=[],wins=[])
                receipt['rows'].append(item); _write(output,receipt)
                print(c['id'],'poses',item['geometry'],'preparations',len(picked),'seconds',item['preparation_s'],flush=True)
                for k,p in enumerate(picked):
                    prefix,unaffected,stamp=physically_prepare(a,old,c,fork,late,p,expect,brisk_flip)
                    note=dict(index=k,pick=plain(p),physically_reached=unaffected,
                              preparation_command_ticks=len(prefix)-fork,
                              actual_q_at_late=a.joints.tolist(),shadow=[])
                    note['late_position_error_max_rad']=float(np.max(abs(a.joints[:5]-p['approach'])))
                    item['trials'].append(note)
                    if not unaffected:
                        note['stopped']=next((e.get('reason') for e in reversed(a.events) if e['kind']=='point'),None)
                        note['offer_difference']=event_difference(a.events,expect)
                        _write(output,receipt)
                        continue
                    try:
                        # This threshold is consumed only by the private native evaluator.
                        a.clearance_margin=private_clearance
                        selected,notes=late_search(a,engine,p,poses,batch,rounds,[400846,c['seed'],k])
                    finally:
                        a.clearance_margin=0.
                    note['shadow']=notes
                    print(c['id'],'preparation',k,p['face'],round(p['p'][1],3),notes,flush=True)
                    if selected:
                        real=physical_finish(a,old,c,leg,prefix,selected[0])
                        note['physical_legal']=real['legal'];note['physical_answered']=real['answered']
                        if real['legal']:
                            real['fork_tick']=fork
                            item['wins'].append(real)
                        if real['answered']:
                            done=True
                    _write(output,receipt)
                    if done:
                        break
                item['cpu_s']=process_time()-tm
                if done and not continue_after:
                    break
        finally:
            a.close()
    receipt['complete']=True
    _write(output,receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT/'long-low.json.gz')
    p.add_argument('--ports',type=int,default=6)
    p.add_argument('--batch',type=int,default=176)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--cases',type=int,nargs='+')
    p.add_argument('--face',type=int,choices=(-1,1))
    p.add_argument('--skip-ports',type=int,default=0)
    p.add_argument('--continue-after',action='store_true')
    p.add_argument('--brisk-flip',action='store_true')
    p.add_argument('--defensive-grid',action='store_true')
    p.add_argument('--private-clearance',type=float,default=0.)
    args=p.parse_args()
    run(args.output,args.ports,args.batch,args.rounds,args.cases,args.face,args.continue_after,args.skip_ports,
        args.brisk_flip,args.defensive_grid,args.private_clearance)
