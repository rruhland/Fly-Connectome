"""Privileged spatial stroke teacher translating carriage-style impacts onto five real joints."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit
from .tt29_partner import _flight_approx
from .tt29_rally_physics import BladeIK, Rollout, NOMINAL_NO_LOAD, measured_launches
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_opponent import arena
from .tt37_opponent_benchmark import BASE, OUT, PLAN, mark
from .tt37_opponent_capacity import ROT


# Near opponent-only teacher has no optic/state interface to the pupil. These values
# describe what stroke could meet the current ball in the arm's physical work volume.
GOAL = np.array([0., 1.265, 1.043, 3.45])
MIRROR_GOAL = GOAL*np.r_[ROT,1.]
DEPTHS = (-1.48,-1.44,-1.40,-1.36,-1.32,-1.28,-1.24,-1.20)
PITCHES = (.04,.14,.24,.36,.50)
DRIVE = ((-.12,0.),(.08,0.),(.34,0.),(.60,0.),(.52,.28),(.35,.52),(.18,.66))


def unit(x):
    return np.asarray(x)/np.linalg.norm(x,axis=-1,keepdims=True)


def body_seeds(ready):
    tapes = [OUT.parent/'tt31'/f for f in ('four-strokes.json','alternating-15.json','face-switch-replay.json')]
    q = [np.asarray(ready)[:5]]
    for path in tapes:
        if path.exists():
            q.extend(np.asarray(x['joints_rad']) for x in _read(path).get('contacts',[]) if x.get('landed'))
    return np.array(q)


def hermite(start, initial, end, speed, eta, times):
    u=np.clip(times/eta,0,1)[:,None]
    positions=(2*u**3-3*u**2+1)*start+(u**3-2*u**2+u)*eta*initial
    positions+=(-2*u**3+3*u**2)*end+(u**3-u**2)*eta*speed
    tail=np.maximum(0.,times[:,None]-eta)
    duration = .125
    ramp = np.minimum(1.,tail/duration)
    return positions+duration*(ramp-.5*ramp**2)*speed


class SpatialTeacher:
    """Use exact observation and native model in the teacher only, not runtime pupil inputs."""

    def __init__(self,a,ready, *, budget=150, static=False, expand=False, refine=False, recover=False, bridge=False):
        self.a=a
        self.ik=BladeIK(a)
        self.replay=Rollout(a,1)
        self.ready=np.asarray(ready)
        self.axes=a._arm_ids[1][2]
        self.drive_cap=a.rate_scale*a._joint_increments[a._arm_ids[1][2]][:5]/a.model.opt.timestep
        self.seed=body_seeds(ready)
        self.seed_xyz=np.asarray([self.ik.pose(q)[0] for q in self.seed])
        self.budget=budget
        self.static=static
        self.expand=expand
        self.refine=refine
        self.recover=recover
        self.bridge=bridge
        self.reception=None
        self.start=-1
        self.stage=0
        self.tape=None
        self.log=[]

    def command(self):
        a=self.a
        if self.tape is not None and a.steps-self.start<len(self.tape):
            # Finish the actuated stroke, including its short follow-through.
            return self.tape[a.steps-self.start]
        if a._last_hitter!=-1 or a.velocity[1]>=-.34 or a._racket_ids[-1] in a._previous_touches:
            if self.recover and a.contacts and a._last_hitter==1:
                # Conduct a real bounded actuator recovery. Never reinstall a posture.
                stride = a._joint_increments[self.axes]*(a.control_dt/a.model.opt.timestep)
                u = np.clip((self.ready-a.data.ctrl[self.axes])/stride,-.58,.58)
                u[-1]=0.
                return u
            return np.zeros(a.action_size)
        if self.reception!=a.rival_contacts:
            self.stage=0
            self.reception=a.rival_contacts
        stage=1+int(a.ball[1]<.70)+int(a.ball[1]<-.12)
        if stage>self.stage and a._geom_ids['table'] not in a._previous_touches:
            self.stage=stage
            self.tape=self.plan()
            self.start=a.steps
            if self.tape is not None:
                return self.tape[0]
        return np.zeros(a.action_size)

    def plan(self):
        a=self.a
        path=a.partner.flight.path(a.ball,a.velocity,a.spin,1,bounced=a._received_bounce,horizon=1.05)
        depths=DEPTHS+(-1.53,-1.57) if self.expand else DEPTHS
        region=path[(path[:,0]>.105)&(path[:,2]<-1.07)&(path[:,2]>-1.68)&
                    (path[:,3]>.875)&(path[:,3]<1.32)&(abs(path[:,1])<.39)]
        waves=[]
        for y in depths:
            if len(region) and min(region[:,2])-.008<=y<=max(region[:,2])+.008:
                pick=region[np.argmin(abs(region[:,2]-y))]
                if all(abs(pick[0]-q[0])>.008 for q in waves):
                    waves.append(pick)
        rep=dict(tick=a.steps,incoming=a.ball.tolist(),velocity=a.velocity.tolist(),reception=self.reception,stage=self.stage,
                 windows=len(waves),spatial_commands=0,translates=0,legal=0)
        self.log.append(rep)
        if not waves:
            return None
        transfer = self.candidates(np.array(waves),rep)
        if transfer is None:
            return None
        goals, proposals, until = transfer
        feedback=np.empty((len(goals),9))
        rates,outcomes=self.replay.trajectories(goals,GOAL,until,feedback=feedback)
        good=np.flatnonzero(np.isfinite(outcomes[:,0]))
        if self.refine and not len(good):
            more, entries=self.local_change(proposals,feedback,goals.shape[1])
            if len(more):
                extra=np.empty((len(more),9))
                trial,new_out=self.replay.trajectories(more,GOAL,until+.028,feedback=extra)
                rep['local_variants']=len(more)
                rep['local_legal']=int(np.sum(np.isfinite(new_out[:,0])))
                rates=np.concatenate([rates,trial])
                outcomes=np.concatenate([outcomes,new_out])
                feedback=np.concatenate([feedback,extra])
                proposals.extend(entries)
                good=np.flatnonzero(np.isfinite(outcomes[:,0]))
        rep['legal']=len(good)
        rep['native_touch_candidates']=int(np.sum(feedback[:,1]>.5))
        rep['closest_m']=float(feedback[:,0].min())
        if not len(good):
            k=int(np.argmax(feedback[:,1]*10+feedback[:,3]-.25*feedback[:,0]))
            rep['closest_departure']=feedback[k].tolist()
            if self.bridge and a.ball[1]>.28 and self.stage<3:
                # A very short motion prefix improves availability before another
                # independent attempt to find a fully legal schedule on the same ball.
                k=int(np.argmin(feedback[:,0]))
                rep['early_motion_prefix']=proposals[k]
                return rates[k,:3]
            return None
        k=int(good[np.argmin(outcomes[good,0])])
        rep['selected']=proposals[k]
        rep['rollout']=outcomes[k].tolist()
        rep['shadow_departure']=feedback[k].tolist()
        contact_time=float(outcomes[k,1])
        # Do not replay the isolated trajectory to its landing. Hand control back
        # after the quick physical stroke so the next opponent flight is observed live.
        return rates[k,:min(len(rates[k]),int(np.ceil((contact_time+.130)/a.control_dt)))]

    def local_change(self, proposals, feedback, frames):
        """Search near successful native motion, on task-space axes rather than joint deltas."""
        a=self.a
        current=a.joints[:5].copy()
        current_dq=a.data.qvel[a._near_dofs][:5].copy()
        jaw=a.data.ctrl[a._arm_ids[1][2]][5]
        hit=feedback[:,1]>.5
        vx,vy,vz=feedback[:,2:5].T
        x,y,z=feedback[:,5:8].T
        time=-y/np.maximum(.1,.89*vy)
        net=z+vz*time-4.9*time**2-.9325
        lateral=np.abs(x+vx*time)
        rank=np.where(hit,np.minimum(.25,net)-.10*lateral+.013*vy,-1.5-feedback[:,0])
        selectable=[]
        for i in np.argsort(-rank):
            p=proposals[i]
            if any(np.linalg.norm(np.subtract(p['center'],proposals[j]['center']))<.027 and
                   np.linalg.norm(np.subtract(p['normal'],proposals[j]['normal']))<.09 for j in selectable):
                continue
            selectable.append(i)
            if len(selectable)==8:
                break
        times=(1+np.arange(frames))*a.control_dt
        categories=((0.,.12,.18,-.012),(0.,-.12,.16,.016),(.042,.09,.25,.010),(-.040,.08,.16,.014),
                    (.022,.24,.15,.006),(-.022,.23,.18,-.008),(0.,.00,.31,.020),
                    (.053,-.07,.20,-.012),(-.054,-.06,.20,.018),(.028,.13,.26,-.006))
        variants, names = [], []
        for i in selectable:
            src=proposals[i]
            center,n,v,qh=map(np.asarray,(src['center'],src['normal'],src['ideal_v'],src['joint']))
            for side,up,pace,wait in categories:
                aim=center+np.array([side,0.,0.])
                normal=unit(n+np.array([-side*1.2,0.,up]))
                q,(at,face,jp,jn)=self.ik.fit(aim,normal,qh)
                if abs((at-aim)@normal)>.038 or np.linalg.norm(at-aim)>.088 or face@normal<.80:
                    continue
                target=v+pace*normal
                rate=np.linalg.lstsq(np.r_[jp,.045*jn,.033*np.eye(5)],np.r_[target,np.zeros(8)],rcond=None)[0]
                rate*= min(1.,NOMINAL_NO_LOAD*.96/max(.05,np.max(abs(rate))))
                moment=src['time']+wait
                lead=src['lead_s']
                command=hermite(current,current_dq,q,rate,moment,times+lead)
                variants.append(np.c_[command,np.full(len(times),jaw)])
                names.append(dict(src,locally_refined=True,center=aim.tolist(),normal=normal.tolist(),
                                  ideal_v=target.tolist(),joint=q.tolist(),qdot=rate.tolist(),time=moment,
                                  jacobian_velocity=(jp@rate).tolist(),pose_gap_m=float(np.linalg.norm(at-aim)),face_dot=float(face@normal)))
        return np.asarray(variants),names

    def candidates(self, waves, rep):
        a=self.a
        q=np.asarray(a.joints)[:5]
        velocity=np.asarray(a.data.qvel[a._near_dofs])[:5]
        ctrl=np.asarray(a.data.ctrl[a._arm_ids[1][2]])
        rows, normals, drives, grouping = [],[],[],[]
        for i,r in enumerate(waves):
            p,v=r[1:4],r[4:7]
            yaw=-.19*p[0]-.12*v[0]
            for bearing in (yaw-.10,yaw,yaw+.10):
                for pitch in PITCHES:
                    n=unit([bearing,1.,pitch-.11*v[2]])
                    tangent=unit(np.array([0.,0.,1.])-n*n[2])
                    group=len(grouping)
                    grouping.append((r,n))
                    for forward,rise in (DRIVE[:1] if self.static else DRIVE):
                        rows.append(r)
                        normals.append(n)
                        drives.append(forward*n+rise*tangent)
        rows,normals,drives=np.asarray(rows),np.asarray(normals),np.asarray(drives)
        m=rows.copy()
        for idx in (1,4,7):
            m[:,idx:idx+3]*=ROT
        p,v,w,centers,_=a.partner.impact.solve(m,normals*ROT,drives*ROT)
        cost=_flight_approx(p,v,w,MIRROR_GOAL)
        centers*=ROT
        rep['spatial_commands']=len(cost)
        rep['isolated_legal']=int(np.sum(np.isfinite(cost)))
        options=np.argsort(cost)
        if np.isfinite(cost).any():
            legal=options[np.isfinite(cost[options])]
        else:
            legal=np.empty(0,dtype=int)
        # Isolated rigid blade is not the arm: preserve a few promising near misses
        # as fine geometry/servo hypotheses even if the guide's ball was under the net.
        failed=options[~np.isfinite(cost[options])]
        score = -v[failed,1] + .75*v[failed,2] if len(failed) else []
        fallback=failed[np.argsort(score)[::-1][:55]]
        candidates=np.r_[legal[:96],fallback]
        if len(candidates)==0:
            return None
        bins=max(2,len(rows)//len(grouping)) if not self.static else 1
        timelines=(1+np.arange(int(np.ceil((waves[:,0].max()+.72)/a.control_dt))))*a.control_dt
        goals,props, cache = [],[],{}
        selected_variants = ((.038,1.0),(.071,1.0),(.053,.66)) if not self.static else ((.035,0.),(.065,0.))
        for j in candidates:
            r,n=rows[j],normals[j]
            pos=centers[j]
            bucket=j//bins
            if bucket not in cache:
                starts=[q]
                nearest=int(np.argmin(np.linalg.norm(self.seed_xyz-pos,axis=1)))
                if np.linalg.norm(self.seed[nearest]-q)>.6:
                    starts.append(self.seed[nearest])
                feasible=[]
                for start in starts:
                    qh,(at,face,Jac,orient)=self.ik.fit(pos,n,start)
                    diff=at-pos
                    if abs(diff@n)>.036 or np.linalg.norm(diff)>.092 or face@n<.79:
                        continue
                    if np.max(abs(qh-q)/self.drive_cap)>r[0]+.105:
                        continue
                    feasible.append((qh,Jac,orient,np.linalg.norm(diff),float(face@n)))
                cache[bucket]=feasible
            for qh,Jac,orient,gap,alignment in cache[bucket]:
                desired=drives[j]
                dq=np.linalg.lstsq(np.r_[Jac,.06*orient,.045*np.eye(5)],np.r_[desired,np.zeros(8)],rcond=None)[0]
                dq *= min(1.,NOMINAL_NO_LOAD*.90/max(.05,np.max(abs(dq))))
                for lead,gain in selected_variants:
                    tail = hermite(q,velocity,qh,dq*gain,r[0],timelines+lead)
                    targets=np.c_[tail,np.repeat(ctrl[5],len(tail))]
                    if len(goals)>=self.budget:
                        break
                    goals.append(targets)
                    props.append(dict(time=r[0],center=pos.tolist(),normal=n.tolist(),ideal_v=desired.tolist(),
                        joint=qh.tolist(),qdot=(dq*gain).tolist(),jacobian_velocity=(Jac@(dq*gain)).tolist(),
                        pose_gap_m=gap,face_dot=alignment,lead_s=lead,isolated_cost=None if not np.isfinite(cost[j]) else float(cost[j])))
            if len(goals)>=self.budget:
                break
        rep['translates']=len(goals)
        return (np.asarray(goals),props,float(waves[:,0].max()+.090)) if goals else None


def run(args):
    d=_read(PLAN)
    feeds=measured_launches(args.feeds)
    raw=_read(BASE)
    sources=d[args.group]
    if args.group=='held_replies':
        sources=sources[::4] if args.stride==0 else sources[::args.stride]
    cards=sources[args.start:args.start+args.count]
    ready=raw['ready'] if args.ready=='raw' else _read(OUT.parent/'tt31'/'centered-ready.json')['ready']
    data=dict(oracle='Exact current ball/spin, grasp model and isolated native rollout are teacher-only.',
              variant=vars(args)|dict(feeds=str(args.feeds),out=str(args.out)),ready=ready,cards=cards,rows=[],
              opening_side=1,source_sha256=_hash(PLAN),feed_sha256=_hash(args.feeds))
    timer=process_time()
    with centered(ready):
        a=arena(robust_contact=True,paddle_grip=args.grasp,clearance_margin=.003)
        try:
            for c in cards:
                shot=feeds[c['id']]
                a.launch(shot,card=c)
                a.partner.reset(seed=c['seed'])
                actor=SpatialTeacher(a,ready,budget=args.budget,static=args.static,expand=args.expand,
                                     refine=args.refine,recover=args.recover,bridge=args.bridge)
                actions=[]
                while True:
                    command=actor.command()
                    _, _, done, cut, _=a.exchange(command,FAR_COMMAND)
                    actions.append(command.tolist())
                    if done or cut or a.legal_returns+a.rival_returns-1>=args.cap:
                        break
                row=dict(shot=shot,opponent_seed=c['seed'],actions=actions,events=a.events,placements=a.point_placements(),
                         score=audit(a.events,opening_side=1),coach=actor.log,motor=a.motor_audit())
                mark(a,row,args.cap)
                data['rows'].append(row)
                _write(args.out,data)
                print(c['id'],row['score']['learner'],row['score']['returns'],
                      [(r['windows'],r['isolated_legal'] if 'isolated_legal' in r else 0,r['translates'],r['legal']) for r in actor.log],
                      len(row['placements']['delivered']),row['physical']['reason'],flush=True)
        finally:
            a.close()
    data['cpu_s']=process_time()-timer
    _write(args.out,data)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('feeds',type=Path)
    p.add_argument('--out',type=Path,default=OUT/'bodycoach-dynamic.json.gz')
    p.add_argument('--group',choices=('practice','held_openings','held_replies'),default='held_replies')
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--count',type=int,default=4)
    p.add_argument('--stride',type=int,default=0)
    p.add_argument('--cap',type=int,default=24)
    p.add_argument('--budget',type=int,default=150)
    p.add_argument('--expand',action='store_true')
    p.add_argument('--static',action='store_true')
    p.add_argument('--refine',action='store_true')
    p.add_argument('--recover',action='store_true')
    p.add_argument('--bridge',action='store_true')
    p.add_argument('--ready',choices=('raw','center'),default='raw')
    p.add_argument('--grasp',choices=('straight','side-on'),default='straight')
    run(p.parse_args())
