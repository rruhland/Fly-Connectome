"""Map many strike times along each live witness flight into collision-checked arm space."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt19_so101 import GRIP_BODY
from .tt29_arm_clearance import ArmClearance
from .tt29_partner import _flight_approx
from .tt29_rally_physics import BladeIK, mujoco
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_bodycoach import MIRROR_GOAL, unit
from .tt37_opponent import arena
from .tt37_opponent_benchmark import BASE, OUT
from .tt37_opponent_capacity import ROT


CLOCK = .024
MOTOR = 3.45
PITCH = (.16,.30,.46,.62)
DRIVES = ((.08,0.),(.40,0.),(.68,.12),(.48,.46),(.24,.66))


def witnessed(events, misses=False):
    far, bounce, table = None, None, None
    out=[]
    for event in events:
        if event['kind']=='separation' and event['collider']=='rival_racket':
            far,bounce,table=event,None,None
        elif event['kind']=='legal_return' and event['side']==-1:
            bounce=event
        elif event['kind']=='separation' and event['collider']=='table' and bounce is not None:
            table=event
        elif event['kind']=='contact' and event['side']==1 and all(x is not None for x in (far,bounce,table)):
            out.append(dict(far=far,bounce=bounce,table=table,near=event))
            far,bounce,table=None,None,None
    if misses and all(x is not None for x in (far,bounce,table)):
        out.append(dict(far=far,bounce=bounce,table=table,near=None))
    return out


class Reach:
    def __init__(self,a,ready,alternate):
        self.a=a
        self.ik=BladeIK(a)
        self.guard=ArmClearance(a.model,tolerance=0.)
        self.d=self.ik.data
        self.ready=np.array([ready,alternate])[:,:5]
        self.grip=a.model.body(GRIP_BODY).id
        self.free=int(a.model.joint('learner_paddle_free').qposadr[0])
        self.weld=a.model.eq_data[a.model.equality('tt19_grasp').id,3:10].copy()
        self.d.qpos[a.model.joint('learner_gripper').qposadr[0]]=ready[-1]
        q=qmc.scale(qmc.Sobol(5,scramble=True,seed=372371).random_base2(13),self.ik.low,self.ik.high)
        self.q=np.concatenate([self.ready,q])
        pose=np.array([np.r_[*self.ik.pose(x)] for x in self.q])
        self.scale=np.array([1,1,1,.10,.10,.10])
        self.index=cKDTree(pose*self.scale)

    def clear(self,q):
        d=self.d
        d.qpos[self.ik.joints]=q
        mujoco.mj_kinematics(self.a.model,d)
        d.qpos[self.free:self.free+3]=d.xpos[self.grip]+d.xmat[self.grip].reshape(3,3)@self.weld[:3]
        mujoco.mju_mulQuat(d.qpos[self.free+3:self.free+7],d.xquat[self.grip],self.weld[3:])
        return not any(value<.002 for _,_,value in self.guard.distances(d,horizon=.003))

    def route(self,q0,q1,steps=10):
        return all(self.clear(q0+s*(q1-q0)) for s in np.linspace(0.,1.,steps))

    def solution(self,p,n,hint):
        query=np.r_[p,n]*self.scale
        nearest=self.index.query(query,k=12)[1]
        seeds=[hint if hint is not None else self.ready[0]]
        for i in nearest:
            s=self.q[i]
            if min(np.max(abs(s-t)) for t in seeds)>.52:
                seeds.append(s)
            if len(seeds)==3:
                break
        valid=[]
        for seed in seeds:
            q,(actual,face,jp,jn)=self.ik.fit(p,n,seed)
            delta=actual-p
            if np.linalg.norm(delta)>.035 or abs(delta@n)>.023 or face@n<.92:
                continue
            if any(np.max(abs(q-other['q']))<.10 for other in valid) or not self.clear(q):
                continue
            valid.append(dict(q=q,p=actual,face=face,jp=jp,jn=jn))
        return valid


def time_windows(scan):
    if not scan:
        return []
    places=sorted(set(round(x['time_from_far_s'],6) for x in scan))
    starts=[places[0]]
    ends=[]
    for a,b in zip(places,places[1:]):
        if b-a>CLOCK*1.6:
            ends.append(a)
            starts.append(b)
    ends.append(places[-1])
    return [list(s) for s in zip(starts,ends,strict=True)]


def evaluate(arm, packet):
    a=arm.a
    shot=packet['table']
    track=a.partner.flight.path(shot['ball'],shot['velocity'],shot['spin'],1,bounced=True,horizon=.78)
    delay=shot['time']-packet['far']['time']
    near=packet['near']
    expected=None if near is None else near['time']-shot['time']
    shadow=track[np.argmin(abs(track[:,0]-expected))] if len(track) and expected is not None else None
    error=None if shadow is None else np.linalg.norm(shadow[1:4]-np.asarray(near['ball']))
    receive=track[np.argmin(abs(track[:,2]+1.1))] if len(track) else None
    usable=track[(track[:,2]<-.88)&(track[:,2]>-1.75)&(track[:,3]>.835)&(track[:,3]<1.32)]
    if len(usable):
        ts=usable[:,0]
        ix=np.unique([np.argmin(abs(ts-t)) for t in np.arange(ts[0],ts[-1]+CLOCK*.35,CLOCK)])
        lines=usable[ix]
    else:
        lines=np.empty((0,10))
    geometries=[]
    proxy=[]
    hint=None
    for k,line in enumerate(lines):
        dt=delay+line[0]
        p,v=line[1:4],line[4:7]
        yaw=-.22*p[0]-.12*v[0]
        for tilt in PITCH:
            norm=unit([yaw,1.,tilt-.11*v[2]])
            center=p-.027*norm
            solves=arm.solution(center,norm,hint)
            for obj in solves:
                q,face,jp,jn=(obj[k] for k in ('q','face','jp','jn'))
                hint=q
                times=np.max(abs(arm.ready-q),axis=1)/MOTOR
                possible=times+.135<=dt
                travel=[bool(b and arm.route(qinit,q)) for b,qinit in zip(possible,arm.ready,strict=True)]
                row=dict(time_from_far_s=float(dt),time_from_bounce_separation_s=float(line[0]),
                         ball_xyz=p.tolist(),normal=face.tolist(),joint=q.tolist(),angle_time_s=times.tolist(),safe_route=travel)
                proposals=[]
                tangent=unit(np.array([0.,0.,1.])-face*face[2])
                for forward,z in DRIVES:
                    wanted=forward*face+z*tangent
                    rates=np.linalg.lstsq(np.r_[jp,.045*jn,.025*np.eye(5)],np.r_[wanted,np.zeros(8)],rcond=None)[0]
                    rates *= min(1.,MOTOR/max(.01,np.max(abs(rates))))
                    actual=jp@rates
                    proposals.append(actual)
                proxy.append((line,face,proposals))
                geometries.append(row)
    if proxy:
        flights=np.repeat(np.asarray([p[0] for p in proxy]),len(DRIVES),axis=0)
        for c in (1,4,7):
            flights[:,c:c+3]*=ROT
        normals=np.repeat(np.asarray([p[1] for p in proxy]),len(DRIVES),axis=0)
        velocities=np.concatenate([p[2] for p in proxy])
        out_p,out_v,out_spin,_,_=a.partner.impact.solve(flights,normals*ROT,velocities*ROT)
        costs=_flight_approx(out_p,out_v,out_spin,MIRROR_GOAL).reshape(-1,len(DRIVES))
        for i,r in enumerate(geometries):
            j=int(np.argmin(costs[i]))
            r['legal_proxy']=bool(np.isfinite(costs[i,j]))
            if r['legal_proxy']:
                r['impact_cost']=float(costs[i,j])
                r['projected_departure_mps']=(out_v[i*len(DRIVES)+j]*ROT).tolist()
            # Independently ask only for physical landing, not the specific
            # useful reply into the desired far receiver window.
            legal_only=None
            for index in range(i*len(DRIVES),(i+1)*len(DRIVES)):
                if out_v[index,1]>=-.25:
                    continue
                window=a.partner.flight.path(out_p[index],out_v[index],out_spin[index],1,horizon=1.15)
                if len(window):
                    legal_only=index
                    break
            r['native_bounce_proxy']=legal_only is not None
    useful=[r for r in geometries if r['legal_proxy']]
    landed=[r for r in geometries if r['native_bounce_proxy']]
    from_raw=[r for r in useful if r['safe_route'][0]]
    from_center=[r for r in useful if r['safe_route'][1]]
    where=near['ball'] if near is not None else receive[1:4].tolist() if receive is not None else shot['ball']
    return dict(actual_contact_m=None if near is None else near['ball'],near_time_s=None if near is None else near['time'],
                x_m=where[0],far_time_s=packet['far']['time'], bounce_m=packet['bounce']['location'],
                position_reconstruction_error_m=None if error is None else float(error),sampled_times=len(lines),
                geometry_slots=time_windows(geometries),legal_proxy_slots=time_windows(useful),
                native_bounce_slots=time_windows(landed),
                native_from_raw=time_windows([r for r in landed if r['safe_route'][0]]),
                native_from_center=time_windows([r for r in landed if r['safe_route'][1]]),
                from_raw=time_windows(from_raw), from_center=time_windows(from_center),records=geometries)


def tally(rows):
    if not rows:
        return {}
    return dict(total=len(rows), geometry=sum(bool(r['geometry_slots']) for r in rows),
                legal_proxy=sum(bool(r['legal_proxy_slots']) for r in rows),
                native_bounce=sum(bool(r['native_bounce_slots']) for r in rows),
                native_from_either=sum(bool(r['native_from_raw'] or r['native_from_center']) for r in rows),
                from_raw=sum(bool(r['from_raw']) for r in rows),
                from_center=sum(bool(r['from_center']) for r in rows),
                from_either=sum(bool(r['from_raw'] or r['from_center']) for r in rows),
                both=sum(bool(r['from_raw'] and r['from_center']) for r in rows))


def run(path, output, limit=None, start=0):
    d=_read(path)
    source=d['rows'][0]
    events=witnessed(source['events'])
    ready=_read(BASE)['ready']
    alt=_read(OUT.parent/'tt31'/'centered-ready.json')['ready']
    clock=process_time()
    source=_hash(path)
    schema='tt37-window-native-2'
    schedule=list(enumerate(events[start:start+limit if limit else None],start))
    rows=[]
    prior_cpu=0.
    if output.exists():
        previous=_read(output)
        if previous.get('schema')==schema and previous.get('source_sha256')==source and previous.get('start')==start:
            rows=previous['rows']
            prior_cpu=previous.get('cpu_s',0.)
            if len(rows)==len(schedule) and previous.get('complete'):
                return
    with centered(ready):
        a=arena(detail='cad',robust_contact=True)
    try:
        body=Reach(a,ready,alt)
        print('safe ready endpoints',*(body.clear(x) for x in body.ready),flush=True)
        for i,e in schedule[len(rows):]:
            r=evaluate(body,e)
            row=dict(leg=i,**r)
            rows.append(row)
            if i%10==0:
                print(i,r['x_m'],r['sampled_times'],len(r['records']),bool(r['from_raw']),bool(r['from_center']),flush=True)
                _write(output,dict(schema=schema,source_sha256=source,start=start,rows=rows,complete=False,
                                   cpu_s=prior_cpu+process_time()-clock))
        by_lane = {k:[r for r in rows if ('left' if r['x_m']<-.075 else 'right' if r['x_m']>.075 else 'middle')==k]
                   for k in ('left','middle','right')}
        _write(output,dict(schema=schema,source_sha256=source,start=start,complete=True,
                          ready_raw=ready,ready_center=alt,scan_clock_s=CLOCK,
                          max_joint_rad_s=MOTOR,source_count=len(events),outputs=tally(rows),
                          by_lane={k:tally(s) for k,s in by_lane.items()},rows=rows,cpu_s=prior_cpu+process_time()-clock))
    finally:
        a.close()


def diagnose(paths, output):
    """Read the articulated pupil's actual encoder state at each later far hit."""
    results=[]
    for path in paths:
        data=_read(path)
        ready=data['ready']
        with centered(ready):
            a=arena(robust_contact=True,paddle_grip=data['variant']['grasp'],clearance_margin=.003)
            geometry=arena(detail='cad',robust_contact=True,paddle_grip=data['variant']['grasp'],clearance_margin=.003)
        try:
            rig=Reach(geometry,ready,ready)
            for c,row in zip(data['cards'],data['rows'],strict=True):
                with centered(ready):
                    a.launch(row['shot'],card=c)
                a.partner.reset(seed=c['seed'])
                stamps=[(0.,a.joints[:5].copy())]
                for action in row['actions']:
                    a.exchange(action,FAR_COMMAND)
                    stamps.append((float(a.data.time),a.joints[:5].copy()))
                if digest_json(a.events)!=digest_json(row['events']):
                    raise ValueError('Body state replay diverged')
                for j, packet in enumerate(witnessed(row['events'],misses=True)[1:],1):
                    prev, current=min(stamps,key=lambda s:abs(s[0]-packet['far']['time']))
                    rig.ready=np.stack([current,np.array(ready[:5])])
                    result=evaluate(rig,packet)
                    result.update(id=c['id'],leg=j,real_reception_q=current.tolist(),state_time_s=prev,
                                  source=path.name,source_sha256=_hash(path))
                    results.append(result)
                    print(c['id'],j,result['x_m'],len(result['records']),bool(result['from_raw']),bool(result['from_center']),flush=True)
        finally:
            a.close()
            geometry.close()
    _write(output,dict(first_ready_field='actual encoder angles at far separation',second_ready_field='nominal start angles',
                       outputs=tally(results),rows=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=OUT/'opponent-capacity-witness.json.gz')
    p.add_argument('--output',type=Path,default=OUT/'bodycoach-windows.json.gz')
    p.add_argument('--limit',type=int)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--diagnose',type=Path,nargs='+')
    args=p.parse_args()
    if args.diagnose:
        diagnose(args.diagnose,args.output)
    else:
        run(args.source,args.output,args.limit,args.start)
