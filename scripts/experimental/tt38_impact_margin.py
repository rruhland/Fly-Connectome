"""Design face and paddle velocity at impact using only a private optical court."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from . import tt33_joint as joint, tt37_motor_clinic as clinic
from .tt4_transfer_trial import _hash, _read, _write
from .tt31_ready import centered
from .tt34_latency import PUPIL
from .tt37_opponent import arena
from .tt37_optical_clinic import OpticalClinic
from .tt38_joint_servo import JointServo
from .tt38_motor_value import MotorValue, OnlineMotorValue
from .tt38_motorcoach import answerable, setup


ROOT = Path(__file__).resolve().parents[2] / 'artifacts'

# Independent angular, height, longitudinal, and cross-court stress pairs.
# The last two pairs cover visible free-flight error and unknown ball rotation.
# SI deltas are position, velocity, then angular velocity in the model frame.
_SHIFT_AXES = np.array([
    (.003,.005,.004,.038,.057,.105,9.,0.,0.),
    (-.002,.009,-.005,-.025,.107,.128,-19.,14.,-11.),
    (.006,-.004,.003,.066,-.055,-.075,28.,-15.,16.),
])
SHIFTS = np.vstack([np.zeros((1,9)), (_SHIFT_AXES[:,None,:]*np.array([-1.,1.])[None,:,None]).reshape(-1,9)])


def normalize(x):
    return x / max(1e-9, np.linalg.norm(x))


def target_launch(p, pace):
    """Nominal flight toward a centered, reachable first bounce under linear drag."""
    length = .74 - p[1]
    drag = .115 * pace
    flight = np.expm1(.115 * length) / drag
    effective = -np.expm1(-drag * flight) / drag
    return np.array([-.86*p[0]/effective, pace, (.78-p[2]+9.81*(flight-effective)/drag)/effective])


def face_and_velocity(p, v, pace, tilt, carry):
    """Use mostly normal impulse rather than relying on tangential ball friction."""
    goal = target_launch(p, pace)
    face = normalize(goal-v)
    face = normalize(face + [0., 0., tilt])
    normal_speed = np.clip((goal+.76*v)@face/1.76, -.28, 1.52)
    # Small vertical transport aids reversal without a sharp wrist torch.
    vertex_speed = normal_speed*face + carry*(np.array([0.,0.,1.])-face[2]*face)
    return face, vertex_speed


def joint_velocity(j, gn, vel):
    return np.clip(np.linalg.lstsq(np.r_[j,.125*gn,.026*np.eye(5)],
                        np.r_[vel,np.zeros(8)],rcond=None)[0],-6.2,6.2)


def reference(q, dq, timeline, eta, lead):
    # A smooth bounded translation holds the face over several possible times
    # of arrival. There is no aggressive angular flick at the collision.
    u = timeline-eta+lead
    span = .113*np.tanh(u/.113)
    return np.c_[q+span[:,None]*dq,np.full(len(timeline),clinic.GRIP)]


def build(surface, obs, eta, q_actual, timeline, legacy, realized):
    x,z,vx,vy,vz=obs[1:6]
    p=np.array([x,surface.depth,z]); v=np.array([vx,vy,vz])
    ik=surface.ik
    rows=[legacy]; descriptions=[dict(family='incumbent')]
    seed=np.array([np.interp(x,surface.xs,q) for q in surface.q.T])
    # Descriptive scan, not a generic high-dimensional random actuator search.
    for pace,tilt,carry in ((3.82,-.035,.03),(4.35,0.,.06),(4.92,-.030,.08),
                            (4.35,.078,.18),(4.92,.065,.19),(5.37,.015,.14)):
        face,vel=face_and_velocity(p,v,pace,tilt,carry)
        q, (at, measured, j, gn) = ik.fit(p-.0245*face,face,.78*seed+.22*q_actual)
        if np.linalg.norm(at-(p-.0245*face))>.038 or measured@face<.94:
            continue
        dq=joint_velocity(j, gn, vel)
        for lead in (.016,.034,.054):
            rows.append(reference(q,dq,timeline,eta,lead))
            descriptions.append(dict(family='normal_sweep',face=face.tolist(),speed=float(vel@face),
                                     pace=pace,tilt=tilt,lead=lead))
    # Stay on the known moving arm trajectory where possible. Change incidence
    # while the solve explicitly cancels unwanted blade-center translation.
    _,n,j,gn=ik.pose(realized,jacobian=True)
    pitch=np.array([0.,-n[2],n[1]])
    pitch_direction=np.linalg.lstsq(np.r_[j,.17*gn,.010*np.eye(5)],
                                    np.r_[np.zeros(3),.17*pitch,np.zeros(5)],rcond=None)[0]
    sweep=joint_velocity(j,gn,n)
    ease = np.clip(timeline / max(.028,.57*eta),0.,1.)
    ease = ease*ease*(3.-2.*ease)
    elapsed=.07*np.tanh((timeline - eta + .027)/.07)
    for tilt,force in ((.065,0.),(-.065,0.),(.125,0.),(-.125,0.),(.05,.29),(.05,-.28),
                       (-.05,.29),(-.05,-.28),(0.,.37),(0.,-.36)):
        d=np.clip(pitch_direction * tilt, -.25,.25)
        gain=np.clip(sweep*force,-3.2,3.2)
        revised=legacy.copy()
        revised[:,:5]+=(d[None,:]+elapsed[:,None]*gain[None,:])*ease[:,None]
        rows.append(revised)
        descriptions.append(dict(family='fixed_center_trim',tilt=tilt,normal_speed_delta=force,
                                 nominal_face=n.tolist(),position_error_m=float(np.linalg.norm(j@d))))
    return np.asarray(rows),descriptions


def quality(out, feedback):
    """Score the legal and answerable interior, not just reaching the ball tip."""
    success=np.isfinite(out[...,0]); hit=feedback[...,1]>.5
    q=np.full(success.shape,-2.8)
    q[hit] = -1.4-.19*np.minimum(8.,clinic.landing_error(feedback[hit]))
    if np.any(success):
        safe=out[success]; f=feedback[success]
        # Penalize low net, touch too far off center, short/wide or long bounces,
        # and a far paddle position outside its comfortable vertical band.
        quality=answerable(safe)
        q[success] = (2.55 - .12*np.minimum(10.,quality)
            - 4.6*np.maximum(.060-safe[:,5],0.) - 3.2*np.maximum(safe[:,12]-.035,0.)
            - .42*np.maximum(.25-f[:,3],0.))
    return q,success


class ImpactMargin:
    def __init__(self, *, stage='margin'):
        self.stage=stage

    @staticmethod
    def output(note, trace=(), tape=None, score=None):
        return dict(tape=tape,trace=list(trace),note=note,score=score,
                    eta=None if score is None else float(score[1]))

    def plan(self, engine, surface, observation, selected, old_score, fit, eta):
        t=engine.twin
        if t.launches or t.card is not None:
            raise ValueError('Impact planner received a physical point instead of a private court')
        if observation is None or not (.055<eta<.232) or engine.latest is None or not np.isfinite(observation[:6]).all():
            return self.output(dict(status='unsupported'))
        obs=np.asarray(observation[:6]); dt=t.control_dt
        n=round((eta+1.06)/dt)
        time=dt*(1+np.arange(n))
        cutoff=eta+.13
        legacy=clinic.schedules(*fit,eta,np.asarray(selected)[None],dt,surface.ready,n)[0]
        logged=np.full((1,n,17),np.nan)
        feedback=np.zeros((1,9))
        _,base=engine.trajectories(legacy[None],clinic.AIM,cutoff,feedback=feedback,traces=logged)
        when=base[0,1] if np.isfinite(base[0,1]) else eta
        near=np.argmin(abs(np.arange(n)*dt-when))
        realized=logged[0,near,6:11]
        if not np.isfinite(realized).all():
            realized=fit[0]
        goals,descriptions=build(surface,obs,eta,np.asarray(engine.previous[:5]),time,legacy,realized)
        f=np.zeros((len(goals),9))
        _,nominal=engine.trajectories(goals,clinic.AIM,cutoff,feedback=f)
        nominal_q,valid=quality(nominal,f)
        trace=[dict(stage='impact_margin_nominal',trials=len(goals)+1,contacts=int(f[:,1].sum()+feedback[:,1].sum()),
                    legal=int(valid.sum()+np.isfinite(base[:,0]).sum()))]
        ranks=np.flatnonzero(valid & (np.arange(len(valid))>0))
        note=dict(status='kept',entered_nominal_legal=old_score is not None,nominal_candidates=int(len(ranks)),
                  incumbent_contact_radius=None if not valid[0] else float(nominal[0,12]))
        if not len(ranks):
            return self.output(note,trace)
        ranks=ranks[np.argsort(-nominal_q[ranks])][:9]
        # Keep at least one member of each physically different motion design.
        for family in ('fixed_center_trim','normal_sweep'):
            candidates=[i for i in np.argsort(-nominal_q) if valid[i] and descriptions[i]['family']==family]
            if candidates and not any(descriptions[k]['family']==family for k in ranks):
                ranks=np.r_[ranks,candidates[0]]
        shortlist=np.r_[0,ranks]
        qmat=[nominal_q[shortlist]]; legal=[valid[shortlist]]
        p,v=t.ball.copy(),t.velocity.copy()
        w=t.data.qvel[t._ball_dof+3:t._ball_dof+6]; refer=w.copy()
        tested=contacts=good=0
        try:
            for d in SHIFTS[1:]:
                t.ball[:]=p+d[:3]; t.velocity[:]=v+d[3:6]; w[:]=refer+d[6:]
                h=np.zeros((len(shortlist),9))
                _,r=engine.trajectories(goals[shortlist],clinic.AIM,cutoff,feedback=h)
                q,b=quality(r,h); qmat.append(q); legal.append(b)
                tested+=len(r); contacts+=int(h[:,1].sum()); good+=int(b.sum())
        finally:
            t.ball[:]=p; t.velocity[:]=v; w[:]=refer
        qmat=np.asarray(qmat); legal=np.asarray(legal)
        # Two low outcomes weight the lateral/pitch tails but cannot make one
        # hard and implausible alternative dominate the entire decision.
        robust=.78*np.mean(qmat,axis=0)+.22*np.mean(np.sort(qmat,axis=0)[:2],axis=0)
        robust[1:] -= .08
        if self.stage=='nominal':
            robust=nominal_q[shortlist]
        i=int(np.argmax(robust)); j=int(shortlist[i])
        trace.append(dict(stage='impact_margin_uncertainty',trials=tested,contacts=contacts,legal=good))
        probability=legal.mean(axis=0)
        nominal_contacts=np.count_nonzero(f[:,1])
        note.update(robust_gain=float(robust[i]-robust[0]),stock_legal_cases=int(legal[:,0].sum()),
                    candidate_legal_cases=int(legal[:,i].sum()),settings=len(SHIFTS),
                    stock_mean=float(robust[0]),candidate_mean=float(robust[i]),
                    candidate=descriptions[j],nominal_contacts=int(nominal_contacts))
        if i==0 or robust[i]-robust[0]<(.37 if valid[0] else .55) or (
            self.stage!='nominal' and (probability[i]<4/7 or (valid[0] and probability[i]<probability[0]))
        ):
            return self.output(note,trace)
        know=nominal[j].tolist()
        frames=min(n,max(1,round((know[1]+.084)/dt)))
        note.update(status='issued',initial_incidence=descriptions[j],feedback=f[j].tolist(),selected_landing=know[3:6])
        return self.output(note,trace,goals[j,:frames].copy(),know)


def games(source, output, stage, count=None):
    prior=_read(source)
    cards=prior['cards'][:count]
    params=clinic.generate(prior['samples'])
    surf=setup(prior['depth'])
    if not np.allclose(surf.ready,prior['ready'],atol=1e-11):
        raise ValueError('Controller and comparison must share the same ready posture')
    joint.RECEIVE_Y=surf.depth
    model=_read(PUPIL)
    bank=prior['bank'][:prior['memory_start']]
    inherited=prior.get('motor_value')
    closed='late_motor_memory' in prior
    online='value_experience' in prior
    impact=ImpactMargin(stage=stage) if stage!='stock' else None
    out={k:v for k,v in prior.items() if k not in ('cards','bank','rows','late_motor_memory','value_experience')}
    out.update(cards=cards,bank=bank,rows=[],source=str(source),source_sha256=_hash(source),stage=stage,
               closed_loop=closed,private_perturbations=SHIFTS.tolist(),completed=False)
    with centered(prior['ready']):
        a=arena()
        eye=OpticalClinic(body=prior['body'],robust=prior['robust'])
        if inherited and inherited!='None' or online:
            start=(_read(Path(inherited)) if inherited and inherited!='None'
                   else dict(x=[np.zeros(11)],y=[[.5,.5]]))
            eye.motor_value=(OnlineMotorValue if online else MotorValue)(start)
        stereo,rig,stride,lo,hi=joint.resources(a)
        servo=JointServo() if closed else None
        try:
            for card, r in zip(cards,prior['rows'][:len(cards)],strict=True):
                before=len(bank)
                row=clinic.point(a,stereo,rig,model,surf,stride,lo,hi,card,r['shot'],bank,params,
                    prior['mode'],prior['cap'],prior['cut'],prior['rounds'],prior['guided'],None,None,
                    prior['refine'],prior['variable'],None,optical_engine=eye,visual=prior['visual'],
                    refresh=prior['refresh'],repair=prior['repair'],camera_stride=prior['camera_stride'],
                    economy=prior['economy'],servo=servo,impact=impact)
                row['memory_size']=[before,len(bank)]
                out['rows'].append(row)
                if servo is not None:
                    out['late_motor_memory']=[dict(state=x.tolist(),delta=u.tolist()) for x,u in servo.memory]
                if online:
                    out['value_experience']=eye.motor_value.examples
                out['completed']=len(out['rows'])==len(cards)
                counts=Counter(x['impact_margin']['status'] for x in row['records'] if 'impact_margin' in x)
                print(card['id'],r['score']['learner'],row['score']['learner'],dict(counts),flush=True)
                _write(output,out)
        finally:
            eye.twin.close(); stereo.close(); a.close()


def compare(source, tested):
    """Pair exact released points and score the issued decisions from physical events."""
    prior, after = _read(source), _read(tested)
    if after['source_sha256']!=_hash(source) or not after['completed'] or prior['cards'][:len(after['cards'])]!=after['cards']:
        raise ValueError('Need complete paired points on the archived, unchanged source')
    pairs=[]
    for card,before,row in zip(after['cards'],prior['rows'][:len(after['cards'])],after['rows'],strict=True):
        bs,rs=before['score'],row['score']
        req=list(zip(before['placements']['requested'],row['placements']['requested']))
        if any(a['target']!=b['target'] for a,b in req) or before['shot']!=row['shot']:
            raise ValueError('Opponent requests or release changed')
        a,b=before['placements']['delivered'],row['placements']['delivered']
        first_ball=bool(a and b and all(a[0][k]==b[0][k] for k in ('target','far_contact','near_bounce','receive_line')))
        if not first_ball and (a or b):
            raise ValueError('First physical opponent offer changed')
        if any(a['time']>b['time'] for a,b in zip(row['events'],row['events'][1:])):
            raise ValueError('Point ledger did not run in physical time order')
        first_change=next((i for i,(u,v) in enumerate(zip(before['actions'],row['actions'])) if u!=v),None)
        if after['stage']=='stock' and (before['actions']!=row['actions'] or before['events']!=row['events']):
            raise ValueError('Stock driver changed stock actions or physical events')
        calls=[j for j in row['records'] if j.get('impact_margin') is not None]
        effects=[]
        for j in calls:
            d=j['impact_margin']
            if d['status']!='issued':
                continue
            start=j['tick']/60.; expected=start+j['score'][1]
            until=expected+.18
            def read_result(r):
                hit=next((e for e in r['events'] if e['kind']=='contact' and e.get('side')==1
                          and start<e['time']<=until),None)
                leave=None if hit is None else next((e for e in r['events'] if e['kind']=='separation'
                    and e.get('collider')=='learner_racket' and hit['time']<=e['time']<=until+.025),None)
                valid=next((x for x in r['score']['shots'] if x['side']==1 and start<x['hit']<=until),None)
                answer=valid is not None and any(x['side']==-1 and x['hit']>valid['landing'] for x in r['score']['shots'])
                return dict(contact=None if hit is None else hit['time'],
                            departure=None if leave is None else leave['velocity'],
                            landing=None if valid is None else valid['landing_xyz'],answered=answer)
            edits=[x['tick'] for x in row['servo'] if x['status']=='corrected' and j['tick']<x['tick']<until*60.]
            effects.append(dict(tick=j['tick'],expected=expected,before=d['stock_legal_cases'],after=d['candidate_legal_cases'],
                family=d['candidate']['family'],twin_departure=d['feedback'][2:5],servo_intervention_ticks=edits,
                same_prior_actions=before['actions'][:j['tick']]==row['actions'][:j['tick']],
                stock=read_result(before),impact=read_result(row)))
        def terminal(row):
            e=next((e for e in reversed(row['events']) if e['kind']=='point'),None)
            return None if e is None else dict(reason=e['reason'],winner=e['winner'])
        trials=[t for r in calls for t in r['search'] if t.get('stage','').startswith('impact_margin')]
        pairs.append(dict(id=card['id'],key=card['key'],first_ball=first_ball,requests_equal=len(req),first_change=first_change,
             stock=dict(own=bs['learner'],total=bs['returns'],cpu_s=before['cpu_s'],terminal=terminal(before)),
             impact=dict(own=rs['learner'],total=rs['returns'],cpu_s=row['cpu_s'],terminal=terminal(row)),
             calls=len(calls),native_trials=sum(t['trials'] for t in trials),selections=effects))
    return dict(source_sha256=_hash(source),tested_sha256=_hash(tested),stage=after['stage'],cards=len(pairs),pairs=pairs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--stage',choices=('stock','nominal','margin'),default='margin')
    p.add_argument('--count',type=int)
    p.add_argument('--compare',type=Path,help='Score this complete tested archive against the source')
    args=p.parse_args()
    if args.compare:
        _write(args.output,compare(args.source,args.compare))
    else:
        games(args.source,args.output,args.stage,args.count)


if __name__=='__main__':
    main()
