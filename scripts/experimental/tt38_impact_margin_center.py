"""Center late geometric strokes on the face using a collision in the private twin."""

import argparse
from copy import copy
from pathlib import Path
from time import process_time
from unittest.mock import patch

import mujoco
import numpy as np

from . import tt33_joint as joint, tt37_motor_clinic as clinic
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import DRAG, LIFT, NOMINAL_NO_LOAD
from .tt31_ready import centered
from .tt34_latency import PUPIL
from .tt37_belief import ArrivalHistory
from .tt37_opponent import arena
from .tt37_optical_clinic import OpticalClinic
from .tt37_visual_inertia import LearnedOptics
from .tt38_impact_margin import ROOT, SHIFTS, normalize, quality
from .tt38_motorcoach import setup


FORKS = {
    'school': ('value-closed', ((3395,1346),(3920,856),(7228,1238),(15294,1287))),
    'independent': ('independent-cold-value', ((13495,787),(11171,1785),(13088,336),(4833,776))),
}


def center_observation(engine, rates, horizon):
    """Rerun one proposed tape in a clone of the unlaunched optical state."""
    t=engine.twin
    if t.launches or t.card is not None:
        raise ValueError('Contact inspection requires an unlaunched optical twin')
    solver=engine.predict
    m=copy(solver.model)
    d=mujoco.MjData(m)
    mujoco.mj_copyData(d,m,t.data)
    low,high=t._joint_low,t._joint_high
    inc=t._joint_increments
    force_range=m.actuator_forcerange[solver.near_axes].copy()
    p,b=t._ball_joint,t._ball_dof
    ball,blade=t._geom_ids['ball'],t._racket_ids[1]
    spin=np.empty(3)
    start=d.time
    for u in rates:
        for _ in range(solver.micro):
            d.ctrl[solver.axes]=np.clip(d.ctrl[solver.axes]+u*inc[solver.axes], low[solver.axes],high[solver.axes])
            other=np.clip((solver.other_home-d.ctrl[solver.other])/solver.micro,-.8*inc[solver.other],.8*inc[solver.other])
            d.ctrl[solver.other]+=other
            v=d.qvel[b:b+3]
            mujoco.mju_rotVecQuat(spin,d.qvel[b+3:b+6],d.qpos[p+3:p+7])
            d.xfrc_applied[t._ball_body,:3]=m.body_mass[t._ball_body]*(-DRAG*np.linalg.norm(v)*v+LIFT*np.cross(spin,v))
            d.xfrc_applied[t._ball_body,3:]=-.18*m.body_inertia[t._ball_body]*spin
            if t.motor_envelope:
                z=d.qvel[solver.near_dofs]/NOMINAL_NO_LOAD
                m.actuator_forcerange[solver.near_axes]=force_range*np.clip(np.c_[1+z,1-z],0.,1.)
            mujoco.mj_step(m,d)
            for touch in d.contact:
                if touch.efc_address<0 or set(touch.geom)!={ball,blade}:
                    continue
                normal=d.geom_xmat[blade].reshape(3,3)[:,2].copy()
                if normal[1]<0:
                    normal=-normal
                off=touch.pos-d.geom_xpos[blade]
                tangent=off-normal*(off@normal)
                return dict(tangent=tangent, radius=float(np.linalg.norm(tangent)), when=float(d.time-start),
                            face=normal,q=d.qpos[engine.positions[:5]].copy())
            if d.time-start>horizon:
                return None
    return None


def reaim(engine,surf,obs,incumbent):
    t=engine.twin
    eta=float(obs[0]); dt=t.control_dt
    old=clinic.schedules(*clinic.geometry(surf,obs),eta,incumbent[None],dt,surf.ready,round((eta+1.13)/dt))
    feedback=np.zeros((1,9))
    rates,s=engine.trajectories(old,clinic.AIM,eta+.12,feedback=feedback)
    hit=center_observation(engine,rates[0],eta+.12)
    if hit is None:
        return None,dict(reason='stock tape did not touch in the camera twin',baseline_feedback=feedback[0].tolist())
    shift_i=hit['tangent']
    if hit['radius']<.039:
        return None,dict(reason='no modeled rim strike to center',radius=hit['radius'])
    center,normal=surf.ik.pose(hit['q'])
    length=len(old[0]); times=(1+np.arange(length))*dt
    candidates=[old[0]]; labels=[dict(move='stock')]
    for step in (.30,.57,.88,1.19,1.48):
        delta=step*shift_i
        if np.linalg.norm(delta)>.074:
            delta*=.074/np.linalg.norm(delta)
        for angle in (-.077,0.,.082):
            nn=normalize(normal+angle*np.array([0.,-normal[2],normal[1]]))
            pose,(at,face,*_)=surf.ik.fit(center+delta,nn,hit['q'])
            if np.linalg.norm(at-(center+delta))>.027:
                continue
            change=np.clip(pose-hit['q'],-.40,.40)
            for fade in (.72,1.15):
                f=np.clip(times/max(.05,hit['when']*fade),0.,1.)
                g=old[0].copy()
                g[:,:5]+=(f*f*(3-2*f))[:,None]*change[None]
                candidates.append(g)
                labels.append(dict(step=step,angle=angle,fade=fade,target_shift_m=delta.tolist(),
                                   pose_fit_error=float(np.linalg.norm(at-(center+delta))),face_fit=float(face@nn)))
    candidates=np.asarray(candidates)
    nf=np.zeros((len(candidates),9))
    _,out=engine.trajectories(candidates,clinic.AIM,eta+.13,feedback=nf)
    score,legal=quality(out,nf)
    score-=16*np.maximum(nf[:,8]-.039,0.)
    note=dict(reason='no robust centered improvement',baseline_feedback=feedback[0].tolist(),radius=hit['radius'],
             optical_twin_touch_s=hit['when'],tangent=shift_i.tolist(),trials=len(candidates)+1,
             predicted_touches=int(nf[:,1].sum()),nominal_legal=int(legal.sum()))
    ix=np.flatnonzero(legal & (np.arange(len(legal))>0) & (nf[:,8]<min(.049,hit['radius']-.008)))
    if not len(ix):
        return None,note
    ids=np.r_[0,ix[np.argsort(-score[ix])[:8]]]
    q=[score[ids]]; valid=[legal[ids]]
    ball,velocity=t.ball.copy(),t.velocity.copy()
    w=t.data.qvel[t._ball_dof+3:t._ball_dof+6]; spin=w.copy()
    try:
        for v in SHIFTS[1:]:
            t.ball[:]=ball+v[:3]; t.velocity[:]=velocity+v[3:6]; w[:]=spin+v[6:]
            h=np.zeros((len(ids),9))
            _,o=engine.trajectories(candidates[ids],clinic.AIM,eta+.13,feedback=h)
            a,b=quality(o,h); a-=16*np.maximum(h[:,8]-.039,0.)
            q.append(a); valid.append(b)
            note['trials']+=len(ids)
    finally:
        t.ball[:]=ball; t.velocity[:]=velocity; w[:]=spin
    scores=np.asarray(q); n=np.asarray(valid).sum(axis=0)
    robust=.75*scores.mean(axis=0)+.25*np.sort(scores,axis=0)[:2].mean(axis=0)
    idx=1+int(np.argmax(robust[1:])); j=int(ids[idx]); option=labels[j]
    note.update(option=option,baseline_legal_cases=int(n[0]),candidate_legal_cases=int(n[idx]),
                candidate_radius=float(nf[j,8]),gain=float(robust[idx]-robust[0]),nominal_departure=nf[j,2:5].tolist(),
                optimized_contact_eta=float(out[j,1]))
    if n[idx]<4 or robust[idx]-robust[0]<.23:
        return None,note
    note['reason']='issued virtual rim-centered schedule'
    return candidates[j,:min(length,round((out[j,1]+.090)/dt))],note


def run(role, output):
    name,forks=FORKS[role]
    source=ROOT/'tt37'/f'{name}.json.gz'
    doc=_read(source)
    joint.RECEIVE_Y=doc['depth']; surf=setup(joint.RECEIVE_Y)
    model=_read(PUPIL)
    targets={c['id']:(c,r) for c,r in zip(doc['cards'],doc['rows'],strict=True)}
    result=dict(role=role,source=str(source.relative_to(ROOT)),source_sha256=_hash(source),forks=list(forks),rows=[])
    with centered(doc['ready']):
        a=arena(); engine=OpticalClinic(body=True)
        stereo,rig,stride,lo,hi=joint.resources(a)
        try:
            for release,tick in forks:
                c,r=targets[release]
                note=next(x for x in r['records'] if x['tick']==tick)
                started=process_time()
                old_events=r['events']; old_actions=r['actions']
                a.launch(r['shot'],card=c); a.partner.reset(seed=c['seed'])
                engine.previous=None; engine.bounced=False
                actor=joint.JointPolicy(model|{'ready':doc['ready']},stride,lo,hi)
                vision=joint.Vision(rig,a.control_dt)
                vision.eye=LearnedOptics(vision.eye,doc['visual'])
                history=ArrivalHistory()
                for k,u in enumerate(old_actions[:tick+1]):
                    q=a.joints.copy()
                    optical=vision.update(k,stereo.capture() if k%2==0 else None,q)
                    optical=history.update(k*a.control_dt,optical)
                    engine.observe(k,vision.eye,q,actor.reference.copy())
                    if k==tick:
                        break
                    a.exchange(u,clinic.FAR)
                    actor.reference=np.clip(actor.reference+np.asarray(u)*stride,lo,hi)
                exact_before=[e for e in old_events if e['time']<=a.data.time+1e-8]
                if digest_json(exact_before)!=digest_json(a.events):
                    raise ValueError('Archived physical prefix failed exact replay')
                if optical is None or not np.allclose(note['context'],np.r_[optical[1:6],q[:5]],atol=1e-8):
                    raise ValueError('Archived camera/encoder context failed exact replay')
                if not engine.sync():
                    raise ValueError('The archived decision no longer has a valid camera estimate')
                moves,private=reaim(engine,surf,optical,np.asarray(note['action']))
                seen=audit(a.events,opening_side=1)
                stats=dict(id=release,tick=tick,previous_own=r['score']['learner'],exact_prefix_own=seen['learner'],
                           past_exact=True,private=private)
                if moves is not None:
                    commands=[]
                    for row in moves:
                        cmd=actor.drive(row,a.joints.copy(),.8)
                        a.exchange(cmd,clinic.FAR)
                        commands.append(cmd.tolist())
                        if a.terminated or a.truncated:
                            break
                    for _ in range(110):
                        later=audit(a.events,opening_side=1)
                        if a.terminated or a.truncated or later['opponent']>seen['opponent']:
                            break
                        u=np.zeros(6)
                        a.exchange(u,clinic.FAR)
                        commands.append(u.tolist())
                    stats.update(new_score=audit(a.events,opening_side=1),events_after=a.events[len(exact_before):],
                                 exact_prefix_ticks=tick,commands=commands,terminated=bool(a.terminated or a.truncated))
                stats['cpu_s']=process_time()-started
                result['rows'].append(stats)
                print(release,private['reason'],{k:private[k] for k in ('radius','candidate_radius','candidate_legal_cases') if k in private},
                      None if 'new_score' not in stats else stats['new_score']['learner'],flush=True)
                _write(output,result)
        finally:
            stereo.close(); engine.twin.close(); a.close()


def radial_audit(path, output):
    """After play, measure the actual signed contact point on full replayed tapes."""
    result=_read(path)
    control_path=ROOT/result['source']
    if _hash(control_path)!=result['source_sha256']:
        raise ValueError('Archived unmodified comparison changed')
    original=_read(control_path)
    cases={c['id']:(c,r) for c,r in zip(original['cards'],original['rows'],strict=True)}
    out=dict(source_sha256=_hash(path),replays=[])
    with centered(original['ready']):
        a=arena()
        try:
            for modified in result['rows']:
                if 'new_score' not in modified:
                    continue
                c,old=cases[modified['id']]
                fork=modified['tick']
                prefix=old['actions'][:fork]
                ledger=[e for e in old['events'] if e['time']<=fork*a.control_dt+1e-8]
                for name,tape,expected in (
                    ('stock',old['actions'],old['events']),('centered',prefix+modified['commands'],ledger+modified['events_after'])
                ):
                    a.launch(old['shot'],card=c); a.partner.reset(seed=c['seed'])
                    contacts=[]
                    r=a._racket_contact
                    def read_contact(side, location, incoming):
                        if side==1 and a.data.time+a.model.opt.timestep>=fork*a.control_dt:
                            blade=a._racket_ids[1]; ball=a._geom_ids['ball']
                            native=[m for m in a.data.contact if m.efc_address>=0 and set(m.geom)=={blade,ball}]
                            if not native:
                                raise ValueError('Recorded racket hit had no native contact')
                            touch=min(native,key=lambda m:np.linalg.norm(m.pos-location))
                            axis=a.data.geom_xmat[blade].reshape(3,3)[:,2]
                            radial=touch.pos-a.data.geom_xpos[blade]
                            radial-=axis*(axis@radial)
                            contacts.append(dict(t=float(a.data.time),radius=float(np.linalg.norm(radial)),
                                                 world_tangent=radial.tolist()))
                        return r(side,location,incoming)
                    with patch.object(a,'_racket_contact',read_contact):
                        for command in tape:
                            a.exchange(command,clinic.FAR)
                    if digest_json(a.events)!=digest_json(expected):
                        raise ValueError('Actual radius replay changed the physical events')
                    out['replays'].append(dict(condition=name,id=c['id'],fork_tick=fork,contacts=contacts,
                        events_sha256=digest_json(a.events),own=audit(a.events,opening_side=1)['learner']))
        finally:
            a.close()
    _write(output,out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('role', choices=(*FORKS,'audit'))
    p.add_argument('output',type=Path)
    p.add_argument('--tested',type=Path)
    args=p.parse_args()
    if args.role=='audit':
        if args.tested is None:
            p.error('Supply an intervention archive for physical radius replay')
        radial_audit(args.tested,args.output)
    else:
        run(args.role,args.output)
