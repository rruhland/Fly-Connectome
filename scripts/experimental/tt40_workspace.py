"""Measure and physically pilot a reachable untucked recovery between live shots."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np
from scipy.optimize import least_squares

from . import tt33_joint as joint, tt37_motor_clinic as clinic
from .tt4_transfer_trial import _hash,_read,_write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt31_ready import centered
from .tt34_latency import PUPIL
from .tt37_belief import ArrivalHistory
from .tt37_bodycoach_windows import Reach
from .tt37_opponent import arena
from .tt37_optical_clinic import OpticalClinic
from .tt37_visual_inertia import LearnedOptics
from .tt38_impact_margin import ROOT, SHIFTS, face_and_velocity, joint_velocity, quality, reference


SOURCE=ROOT/'tt37'/'value-closed.json.gz'
DEPTHS=(-1.36,-1.45,-1.55,-1.65,-1.72)
SHOULDER=-1.58


def fit_branch(ik,p,n,seed,limit=None):
    lo=ik.low.copy(); hi=ik.high.copy()
    if limit is not None:
        lo[1]=max(lo[1],limit)
    q0=np.clip(seed,lo+1e-6,hi-1e-6)
    def f(q):
        center, normal=ik.pose(q)
        return np.r_[center-p,.096*(normal-n),.010*(q-q0)]
    def j(q):
        _,_,linear,angular=ik.pose(q,jacobian=True)
        return np.r_[linear,.096*angular,.010*np.eye(5)]
    ans=least_squares(f,q0,jac=j,bounds=(lo,hi),max_nfev=23,ftol=1e-4,gtol=1e-5)
    q=ans.x; center,face,J,G=ik.pose(q,jacobian=True)
    return q,center,face,J,G


def atlas(output):
    d=_read(SOURCE); home=np.asarray(d['ready']); targets=[]
    with centered(home):
        art=arena(detail='cad')
        rig=Reach(art,home,home)
        try:
            qsample=rig.q[rig.q[:,1]>SHOULDER]
            positions=np.array([np.r_[*rig.ik.pose(q)] for q in qsample])*rig.scale
            for x in (-.23,0.,.23):
                for z in (.89,.94,.99,1.075,1.15):
                    for y in DEPTHS:
                        b=np.array([x,y,z]); modes={'tucked':[],'untucked':[]}
                        for variant,limit in (('tucked',None),('untucked',SHOULDER)):
                            for pitch in (.34,.58,.84):
                                n=np.array([-.24*x,np.sqrt(1.-pitch*pitch),pitch]); n/=np.linalg.norm(n)
                                p=b-.023*n
                                near=np.argsort(np.linalg.norm(positions-np.r_[p,n]*rig.scale,axis=1))[:32]
                                seeds=[home[:5]]
                                for i in near:
                                    if min(np.max(abs(qsample[i]-q)) for q in seeds)>.47:seeds.append(qsample[i])
                                    if len(seeds)>=4:break
                                for guess in (seeds[:1] if limit is None else seeds):
                                    q,a,face,*_=fit_branch(rig.ik,p,n,guess,limit)
                                    err=float(np.linalg.norm(a-p)); aligned=float(face@n)
                                    if err>.047 or abs((a-p)@face)>.023 or aligned<.94 or not rig.clear(q):continue
                                    if any(np.max(abs(q-past['q']))<.10 for past in modes[variant]):continue
                                    modes[variant].append(dict(q=q.tolist(),center_error=err,face=aligned,normal=n.tolist()))
                        targets.append(dict(ball=b.tolist(),**modes))
            endpoints=[np.asarray(r['q']) for t in targets for r in t['untucked']]
            headers=dict(shoulder_floor=SHOULDER,ready=home.tolist(),control_sha256=_hash(SOURCE),targets=targets)
            sets={}
            for group,zrule in (('low',lambda z:z<1.0),('high',lambda z:z>=1.075)):
                rows=[t for t in targets if zrule(t['ball'][2])]
                sets[group]=dict(targets=len(rows),tucked=sum(bool(t['tucked']) for t in rows),
                                 untucked=sum(bool(t['untucked']) for t in rows))
            if not endpoints:
                _write(output,headers|dict(coverage=sets,hubs=[])); return
            # Seek central connected hubs that preserve the nominal higher shot family too.
            low_targets=[t for t in targets if t['ball'][2]<1. and t['untucked']]
            high_targets=[t for t in targets if t['ball'][2]>=1.075 and (t['tucked'] or t['untucked'])]
            candidates=endpoints+[np.median(endpoints,axis=0)]
            candidates.extend(.66*np.asarray(x)+.34*y for x in endpoints for y in [home[:5],np.median(endpoints,axis=0)])
            incoming=[]
            for q in candidates:
                blade,_=rig.ik.pose(q)
                if q[1]<SHOULDER or abs(blade[0])>.21 or not .88<blade[2]<1.13 or not rig.clear(q):continue
                if not rig.route(home[:5],q,steps=15):continue
                if any(np.max(abs(q-e['q']))<.10 for e in incoming):continue
                def distance(t):
                    return min(np.max(abs(q-e['q'])) for e in t['untucked']+t['tucked'])
                low=np.array([distance(t) for t in low_targets]); high=np.array([distance(t) for t in high_targets])
                if not len(high):continue
                # Penalize slow paths to both groups. A hub that only gains access
                # to low balls at the cost of high ones does not win by construction.
                score=float((np.quantile(low,.8) if len(low) else 0.)+1.1*np.quantile(high,.8)+.12*np.linalg.norm(q-home[:5]))
                incoming.append(dict(q=q.tolist(),blade=blade.tolist(),score=score,
                                     low_rad=None if not len(low) else float(np.quantile(low,.8)),high_rad=float(np.quantile(high,.8))))
            hubs=sorted(incoming,key=lambda e:e['score'])[:9]
            _write(output,headers|dict(coverage=sets,hubs=hubs))
            print(sets,len(hubs),hubs[:1],flush=True)
        finally:
            art.close()


class ContactPath:
    def __init__(self,engine,cal,guard):
        self.t=engine; self.guard=guard
        self.k=guard.ik
        self.hubs=[np.asarray(h['q']) for h in cal['hubs'][:3]]

    def find(self):
        e=self.t; t=e.twin
        if t.launches or t.card is not None or not e.sync():
            return None,dict(status='no independent optical state')
        track=t.partner.flight.path(t.ball,t.velocity,t.spin,1,bounced=t._received_bounce,horizon=.56)
        windows=track[(track[:,0]>.09)&(track[:,0]<=.46)&(track[:,3]>.835)&(track[:,3]<1.27)]
        ref=e.reference[:5]; q0=e.previous[:5]
        dt=t.control_dt; n=round(1.57/dt); time=(1+np.arange(n))*dt
        g=[]; meta=[]
        for depth in (-1.44,-1.51,-1.58,-1.65,-1.71):
            if not len(windows) or not windows[:,2].min()-.009<depth<windows[:,2].max()+.009:continue
            stamp,x,y,z,*v0=windows[np.argmin(abs(windows[:,2]-depth))][:7]
            p=np.array([x,y,z]); v=np.asarray(v0)
            for pace,angle,lift in ((3.9,.04,.27),(4.4,.10,.36),(4.9,-.04,.24),(4.4,.19,.45)):
                face,speed=face_and_velocity(p,v,pace,angle,lift)
                for seed,limit in ((q0,None),(ref,None),*((q,SHOULDER) for q in self.hubs[:2])):
                    q,near,norm,J,G=fit_branch(self.k,p-.024*face,face,seed,limit)
                    if np.linalg.norm(p-.024*face-near)>.037 or norm@face<.935:continue
                    if np.max(abs(q-q0))>7.0*(stamp+.024) or not self.guard.route(q0,q,steps=6):continue
                    vstrike=joint_velocity(J,G,speed)
                    for lead in (.021,.043):
                        g.append(reference(q,vstrike,time,stamp,lead))
                        meta.append(dict(depth=float(y),height=float(z),geometry_eta=float(stamp),pace=pace,
                                         shoulder=float(q[1]),lead=lead))
        if not g:return None,dict(status='no reachable pose',candidates=0)
        goals=np.asarray(g); fb=np.zeros((len(g),9)); cutoff=.52
        _,out=e.trajectories(goals,clinic.AIM,cutoff,feedback=fb)
        v,ok=quality(out,fb)
        valid=np.flatnonzero(ok)
        report=dict(status='no legal output',candidates=len(g),contacts=int(fb[:,1].sum()),nominal_legal=len(valid))
        if len(valid)==0:return None,report
        ids=valid[np.argsort(-v[valid])[:10]]
        p=t.ball.copy(); vel=t.velocity.copy(); scores=v[ids].copy(); legal=np.ones(len(ids),int)
        try:
            for d in SHIFTS[1:3]:
                t.ball[:]=p+d[:3]; t.velocity[:]=vel+d[3:6]
                f=np.zeros((len(ids),9))
                _,r=e.trajectories(goals[ids],clinic.AIM,cutoff,feedback=f)
                score,okay=quality(r,f); scores+=score; legal+=okay
        finally:
            t.ball[:]=p; t.velocity[:]=vel
        idx=int(np.argmax(scores)); j=int(ids[idx]); robust=int(legal[idx])
        report.update(selection=meta[j],settings=robust,predicted_touch=float(out[j,1]),predicted_velocity=fb[j,2:5].tolist())
        if robust<2:return None,report|dict(status='not robust')
        return goals[j,:round((out[j,1]+.085)/dt)],report|dict(status='issued')


def pilot(calibration, output, which):
    d=_read(SOURCE); cal=_read(calibration)
    if cal['control_sha256']!=_hash(SOURCE) or not cal['hubs']:
        raise ValueError('A protected high/low workspace calibration is required')
    chosen={'low':(15294,13),'deep':(6003,9),'high':(3920,4)}[which]
    c,r=next((c,r) for c,r in zip(d['cards'],d['rows'],strict=True) if c['id']==chosen[0])
    own=[s for s in r['score']['shots'] if s['side']==1]
    # This is an archived selection of a live transition, not an input to the actor.
    old=own[chosen[1]-1]
    fork=round((old['hit']+.18)*60)
    initial=[e for e in r['events'] if e['time']<=fork/60+1e-8]
    model=_read(PUPIL); joint.RECEIVE_Y=d['depth']; home=np.asarray(d['ready'])
    result=dict(source_sha256=_hash(SOURCE),calibration_sha256=_hash(calibration),card=c,label=which,
                fork_tick=fork,ready=home.tolist(),previous_point_own=r['score']['learner'],rows=[])
    hub=np.r_[cal['hubs'][0]['q'],home[-1]]
    with centered(home):
        a=arena(); cad=arena(detail='cad'); engine=OpticalClinic(body=True)
        guard=Reach(cad,home,home); search=ContactPath(engine,cal,guard)
        renderer,rig,stride,lo,hi=joint.resources(a)
        try:
            for mode in ('stock_prep','untucked_prep'):
                start=process_time(); a.launch(r['shot'],card=c); a.partner.reset(seed=c['seed'])
                engine.previous=None; engine.bounced=False
                actor=joint.JointPolicy(model|{'ready':home.tolist()},stride,lo,hi)
                eye=joint.Vision(rig,a.control_dt); eye.eye=LearnedOptics(eye.eye,d['visual'])
                history=ArrivalHistory(); actions=[]; notes=[]; moved=[]
                ready=False; tape=None; issued=None; next_review=fork; before=None; first_inbound=None
                for tick in range(min(a.max_steps,fork+225)):
                    now=tick*a.control_dt; q=a.joints.copy()
                    obs=history.update(now,eye.update(tick,renderer.capture() if tick%2==0 else None,q))
                    engine.observe(tick,eye.eye,q,actor.reference.copy())
                    if tick==fork:
                        if digest_json(initial)!=digest_json(a.events):raise ValueError('Archived approach changed before intervention')
                        before=audit(a.events,opening_side=1)
                        branch_ok=guard.route(q[:5],hub[:5],steps=16) and guard.route(actor.reference[:5],hub[:5],steps=13)
                    raw=eye.eye.latest
                    if tick>=fork and raw is not None and now-raw.time<.086:
                        pos,vel=raw.advance(now)
                        if vel[1]>.85 and pos[1]>-.55:ready=True
                        if ready and vel[1]<-.9 and first_inbound is None:
                            first_inbound=dict(tick=tick,q=q.tolist(),reference=actor.reference.tolist())
                    def fallback():
                        if moved:
                            if obs is not None and .035<obs[0]<.60:
                                x,z,vx,_,vz=obs[1:6]
                                face=np.array([-.18*x-.085*vx,1.,.29-.095*vz]); face/=np.linalg.norm(face)
                                p=np.array([x,d['depth'],z])-.025*face
                                center,_,_,J,G=fit_branch(guard.ik,p,face,home[:5])
                                drift=joint_velocity(J,G,face)
                                target=np.r_[center-.072*drift,home[-1]]
                                if guard.route(q[:5],target[:5],steps=7):return actor.drive(target,q,.60)
                            return np.zeros(6)
                        if tick<len(r['actions']):
                            u=np.asarray(r['actions'][tick]); actor.reference=np.clip(actor.reference+stride*u,lo,hi)
                            return u
                        return actor.drive(home,q,.8)
                    if tape is not None:
                        elapsed=tick-issued
                        des=tape[elapsed] if elapsed<len(tape) else home
                        cmd=actor.drive(des,q,.8)
                    elif tick>=next_review and first_inbound is not None and obs is not None and .09<obs[0]<.335:
                        tape,note=search.find(); notes.append(dict(tick=tick,**note)); next_review=tick+5
                        if tape is not None:
                            issued=tick; cmd=actor.drive(tape[0],q,.8)
                        else:cmd=fallback()
                    elif mode=='untucked_prep' and tick>=fork and branch_ok and ready and first_inbound is None:
                        cmd=actor.drive(hub,q,.56); moved.append(tick)
                    else:cmd=fallback()
                    a.exchange(cmd,clinic.FAR); actions.append(np.asarray(cmd).tolist())
                    if a.terminated or a.truncated:break
                    if tick>fork and before is not None:
                        score=audit(a.events,opening_side=1)
                        # Deliberately stop after the queried next stroke and its reply.
                        if score['learner']>chosen[1] and score['opponent']>before['opponent']+1:break
                score=audit(a.events,opening_side=1)
                result['rows'].append(dict(mode=mode,ready_route_checked=bool(branch_ok),outbound_ticks=moved,first_inbound=first_inbound,
                    selected_tick=issued,notes=notes,actions=actions,events=a.events,score=score,placements=a.point_placements(),
                    shot=r['shot'],terminated=bool(a.terminated or a.truncated),cpu_s=process_time()-start))
                print(which,mode,len(moved),score['learner'],score['returns'],issued,flush=True)
                _write(output,result)
        finally:
            renderer.close(); engine.twin.close(); cad.close(); a.close()


def finish(saved,output):
    """Resolve slow flights after the first bounded receipt without a new point."""
    d=_read(saved)
    if d['source_sha256']!=_hash(SOURCE):raise ValueError('Archived physical control changed')
    source=_read(SOURCE)
    c=d['card']; old=next(r for a,r in zip(source['cards'],source['rows'],strict=True) if a==c)
    chosen={'low':13,'deep':9,'high':4}[d['label']]
    prefix=[e for e in old['events'] if e['time']<=d['fork_tick']/60+1e-8]
    far_before=audit(prefix,opening_side=1)['opponent']
    home=np.array(d['ready']); model=_read(PUPIL)
    out={k:v for k,v in d.items() if k!='rows'}|dict(rows=[],extended_from_sha256=_hash(saved))
    with centered(home):
        a=arena()
        ax=a._arm_ids[1][2]
        dt=a.control_dt
        stride=a._joint_increments[ax]*(dt/a.model.opt.timestep)
        lo,hi=a._joint_low[ax],a._joint_high[ax]
        try:
            for r in d['rows']:
                a.launch(r['shot'],card=c); a.partner.reset(seed=c['seed'])
                actor=joint.JointPolicy(model|{'ready':home.tolist()},stride,lo,hi)
                for u in r['actions']:
                    a.exchange(u,clinic.FAR)
                    actor.reference=np.clip(actor.reference+stride*np.asarray(u),lo,hi)
                if digest_json(r['events'])!=digest_json(a.events):raise ValueError('Continuation prefix failed physical replay')
                actions=list(r['actions']); tail=[]
                for _ in range(120):
                    current=audit(a.events,opening_side=1)
                    if a.terminated or a.truncated or (current['learner']>chosen and current['opponent']>far_before+1):break
                    # The proposed strike and its follow-through already ended.
                    u=actor.drive(home,a.joints.copy(),.8)
                    a.exchange(u,clinic.FAR); tail.append(u.tolist())
                score=audit(a.events,opening_side=1)
                print(d['label'],r['mode'],score['learner'],score['returns'],len(tail),flush=True)
                out['rows'].append(r|dict(actions=actions+tail,events=a.events,score=score,placements=a.point_placements(),
                    prefix_verified=True,uninterrupted_suffix_ticks=len(tail),terminated=bool(a.terminated or a.truncated)))
        finally:
            a.close()
    _write(output,out)


def clearance(saved,output):
    """Full reachable-time CAD replay of the actually driven preparation."""
    from .tt37_opponent_audit import CourtClearance, contact_observer, guarded, verify_clearance
    d=_read(saved); c=d['card']; records=[]
    with centered(d['ready']):
        a=arena(); cad=arena(detail='cad')
        try:
            for r in d['rows']:
                a.launch(r['shot'],card=c); a.partner.reset(seed=c['seed'])
                guarded(a,CourtClearance(cad.model,articulated=True))
                contacts=[]
                with contact_observer(a,contacts):
                    for u in r['actions']:a.exchange(u,clinic.FAR)
                if any(digest_json(v)!=digest_json(r[k]) for k,v in (
                    ('events',a.events),('score',audit(a.events,opening_side=1)),('placements',a.point_placements())
                )):raise ValueError('Full CAD replay changed archived actions or score')
                verify_clearance(a)
                records.append(dict(mode=r['mode'],returns=r['score']['returns'],native_contacts=len(contacts),
                    clearance=a.clearance_result(),events_sha256=digest_json(a.events)))
        finally:
            a.close(); cad.close()
    _write(output,dict(source_sha256=_hash(saved),records=records))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('atlas','low','deep','high','finish','clearance'))
    p.add_argument('output',type=Path)
    p.add_argument('--atlas',type=Path)
    p.add_argument('--saved',type=Path)
    args=p.parse_args()
    if args.mode=='atlas':atlas(args.output)
    elif args.mode in ('finish','clearance'):
        if args.saved is None:p.error('Supply the bounded pilot receipt')
        (finish if args.mode=='finish' else clearance)(args.saved,args.output)
    elif args.atlas is None:p.error('Supply a completed workspace atlas')
    else:pilot(args.atlas,args.output,args.mode)
