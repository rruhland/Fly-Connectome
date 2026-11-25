"""Acquire and test an optically phased forecourt midline stroke from its physically reached body."""
import argparse
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from . import tt37_opponent as opponent
from . import tt51_low_reverse_curriculum as optical
from . import tt54_midline as native
from . import tt54_midline_posture as posture
from .tt4_transfer_trial import _hash, _read
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import BladeIK
from .tt37_opponent_audit import CourtClearance, guarded
from .tt39_carry_joints import Carried
from .tt40_long_low import same_trace
from .tt44_paired_credit import State

ROOT=native.ROOT
PACKET=ROOT/'forecourt-rgb.json.gz'
PLAN=ROOT/'forecourt-protocol.json'
RESULT=ROOT/'forecourt-course.json.gz'
STD=opponent.draw
ID=4214
NBR=((-0.012,0.,0.,0.),(.012,0.,0.,0.),(0.,0.,-.006,0.),(0.,0.,.006,0.))
ENGINE_FIELDS=('tick','reference','rate','previous','bounced')


def teacher():
    data=_read(posture.RESULT)
    row=next(r for r in data['rows'] if r['case']['id']==ID)
    if row['win'] is None:
        raise ValueError('No independently CAD-verified midline source')
    live=row['routes'][row['win']]['physical']
    if not live['answered'] or live['full_cad']['penetration_samples']:
        raise ValueError('No independently CAD-verified midline source')
    old=_read(Path(row['case']['source']))
    return row,live,old['ready'],old['rows'][row['case']['index']]


def cue_prototype(log,template):
    at=template['source_phase_tick']
    candidates=[r for r in log if r['tick']<=at and r['x'] is not None and r['x'][4]<-.9 and r['bounce']]
    if not candidates:
        raise ValueError('No observable uninterrupted postbounce forecourt track')
    p=np.asarray(candidates[-1]['x'])
    elapsed=(-1.10-p[1])/(.98*p[4])
    if not -.014<elapsed<.095:
        raise ValueError('Optical context extrapolation is unsupported')
    return np.r_[p[0]+p[3]*elapsed,p[2]+p[5]*elapsed-4.905*elapsed**2,p[3:5],p[5]-9.81*elapsed]


def capture():
    row,live,ready,old=teacher();c=row['card']; leg=row['case']['leg']
    root=row['fork_tick']; cut=row['receiving_tick']
    begin=max(0,root-36)
    with mpc.centered(ready):
        a=mpc.arena(); camera,rig,stride,lo,hi=mpc.resources(a); axes=a._arm_ids[1][2]
        try:
            a.launch(live['shot'],card=c);a.partner.reset(seed=c['seed'])
            eye=mpc.Vision(rig,a.control_dt,raw=True);eye.eye=mpc.LearnedOptics(eye.eye,course.OPTIC)
            log=[]
            for t,u in enumerate(live['actions']):
                q=a.joints.copy();ref=a.data.ctrl[axes].copy()
                x=mpc.observed(eye,t,camera.capture() if t%2==0 and t>=begin else None,q)
                b=bool(eye.eye.latest and eye.eye.latest.bounced)
                if t>=begin:
                    log.append(dict(tick=t,x=None if x is None else x.tolist(),q=q.tolist(),ref=ref.tolist(),
                        next_ref=np.clip(ref+np.asarray(u)*stride,lo,hi).tolist(),bounce=b))
                a.exchange(u,mpc.FAR_COMMAND)
            if digest_json(a.events)!=digest_json(live['events']):
                raise ValueError('Camera extraction altered the physical teacher')
            # Preparation and finishing are learned separately. Extraction
            # uses the ordinary direction change plus observed ball-to-blade
            # proximity. It never receives the actual near-contact tick.
            learned=optical.extract(log,BladeIK(a),'forecourt-4214',cut-8)
            if learned is None:
                raise ValueError('The existing RGB and encoder labeler cannot support this contact')
            learned['p']=cue_prototype(log,learned)
            learned['case']=dict(release=ID,leg=leg)
            optical.save(PACKET,dict(teacher_sha256=digest_json(live['actions']),templates=[learned],ready=ready,
                acquisition=dict(fork=root,cut=cut,observations=log,
                    reason='Normal cameras and proprioception along the scored actual commands. FK and optical phase only in the template. True reward authorizes creation.')))
            print('RGB',learned['source_phase_tick'],learned['depth'],learned['minimum_camera_body_gap'],learned['p'],flush=True)
        finally:
            camera.close();a.close()


def plan():
    row,live,ready,old=teacher()
    packet=_read(PACKET)
    if packet['teacher_sha256']!=digest_json(live['actions']):
        raise ValueError('Different earned optical exit rehearsal')
    leg=row['case']['leg'];c=row['card']
    request=live['placements']['requested'][leg]
    anchor=int(np.floor(request['time_s']/mpc.DT+1e-7))
    rng=np.random.default_rng(541903)
    held=[]
    while len(held)<4:
        seed=int(rng.integers(1,2**30)); target=STD(dict(c,reply_seed=seed),leg,'random',None)[0]
        if abs(target[0]-request['target'][0])<.045 and abs(target[2]-request['target'][2])<.027:
            held.append(dict(seed=seed,target=list(target)))
    broad=[int(q) for q in rng.integers(1,2**30,4)]
    optical.save(PLAN,dict(teacher_sha256=packet['teacher_sha256'],packet_sha256=_hash(PACKET),
        id=ID,leg=leg,anchor=anchor,cut=row['receiving_tick'],seed_target=request['target'],neighbor_deltas=NBR,
        nearest_natural_streams=held,broad_streams=broad,
        role='Stock and optical motor recall share the physically reached prior body and actual opponent delivery. No replayed future phase after handoff. Exact acquisition then four disclosed local training perturbations followed by four independently sampled full TT37 reply targets in a predeclared nearby-request rectangle and four untouched broad random streams when the prior stage succeeds. Opponent physics is unchanged.'))


class Forecourt(season.Player):
    def begin(self):
        super().begin()
        self.stroke=None
        self.struck_at=-99
        self.evidence=[]
        self.templates=None

    def forecourt_command(self,tick,eye,x,q,ref,home):
        # Only observed flight and measured body/reference enter this path.
        # Native landmark predictions may rank optical proposals but cannot
        # provide their training labels or any true future time.
        if self.stroke is not None and 0<=tick-self.struck_at<len(self.stroke):
            self.engine.observe(tick,eye,q,ref)
            return self.stroke[tick-self.struck_at]
        base=home
        if x is not None and x[4]<-.9:
            desired, conf=self.raw.get(x,q)
            if desired is not None:
                base=home+conf*(desired-home)
        stock=super().step(tick,eye,x,q,ref,base)
        if (self.templates is None or self.stroke is not None or x is None or x[4]>-.9
                or not -.97<x[1]<-.47 or not self.engine.bounced or not self.engine.sync()):
            return stock
        a=self.engine.twin
        path=a.partner.flight.path(a.ball,a.velocity,a.spin,1,bounced=True,horizon=.44)
        if not len(path):
            return stock
        z=path[np.argmin(abs(path[:,2]+1.10))]
        f=np.r_[z[1],z[3:7]]
        if abs(f[0])>.105:
            return stock
        proposed=self.templates.proposals(f,path,x)
        good=[(name,g) for name,g in proposed if np.max(abs(g[0,:5]-q[:5]))<.52]
        if not good:
            return stock
        names,goals=map(list,zip(*good))
        loss,ok,_,y,_=self.trial(goals,long=True)
        vote=np.zeros(len(ok),dtype=int)
        if ok.any():
            vote[ok]=self.uncertainty([g for j,g in enumerate(goals) if ok[j]],loss[ok])[0]
        suited=[(np.linalg.norm((f-r['p'])/[.13,.073,.75,.84,.88]),r) for r in self.templates.rows]
        distance,r=min(suited,key=lambda q:q[0]);options=[j for j,name in enumerate(names) if name==r['name']]
        eta=float(path[np.argmin(abs(path[:,2]-r['depth'])),0])
        note=dict(tick=tick,optical=x.tolist(),forecast=f.tolist(),n=len(goals),native_votes=vote.tolist(),
                  public_eta_s=eta,best_template=distance,issued=False)
        self.evidence.append(note)
        if (vote>0).any():
            j=int(np.argmin(np.where(vote>0,loss,np.inf)))
        elif distance<.65 and .045<eta<.16 and len(options)==3:
            # Literal optical retiming (middle zero-phase-shift candidate).
            # A bounded practice comparison also tests learned motor memory
            # when the independent model rejects all close alternatives.
            j=options[1]
        else:
            return stock
        self.stroke=np.array(goals[j]);self.struck_at=tick
        self.stroke=self.stroke[:min(len(self.stroke),int(np.ceil(eta/a.control_dt))+5)]
        note.update(issued=True,name=names[j],choice=j,fallback=not bool(vote[j]))
        return self.stroke[0]


def body_snapshot(engine):
    state=mujoco.MjData(engine.body_model)
    mujoco.mj_copyData(state,engine.body_model,engine.body_state)
    return state,{k:deepcopy(getattr(engine,k)) for k in ENGINE_FIELDS}


def restore_body(engine,saved,eye):
    state,values=saved
    mujoco.mj_copyData(engine.body_state,engine.body_model,state)
    for key,value in values.items():
        setattr(engine,key,deepcopy(value))
    engine.tracker=eye.eye
    engine.latest=eye.eye.latest


def one(a,res,actor,root_state,camera_state,engine_state,c,live,p,template,seed,target,stage,index,check,*,family=None):
    camera,rig,stride,lo,hi=res;axes=a._arm_ids[1][2];leg=p['leg'];cut=p['cut']
    root_state.restore(a);a.card=dict(c,reply_seed=seed)
    eye=deepcopy(camera_state)
    restore_body(actor.engine,engine_state,eye)
    for t in range(p['anchor'],cut):
        q=a.joints.copy();mpc.observed(eye,t,camera.capture() if t%2==0 else None,q)
        actor.engine.observe(t,eye.eye,q,a.data.ctrl[axes].copy())
        _,_,end,cap,_=a.exchange(live['actions'][t],mpc.FAR_COMMAND)
        if end or cap:
            return dict(index=index,stage=stage,status='ended_during_fixed_posture',variants={})
    common=State(a);q0=a.joints.copy(); home=a.data.ctrl[axes].copy(); states={}
    engine_common=body_snapshot(actor.engine)
    for mode in ('stock','learned') if family is None else ('stock','learned','family'):
        common.restore(a); e=deepcopy(eye)
        actor.begin()
        # The isolated body was warmed on encoder and issued-reference
        # history. No native dynamic state or future teacher pose is copied.
        restore_body(actor.engine,engine_common,e)
        actor.templates=template if mode=='learned' else family if mode=='family' else None
        guarded(a,check);began=float(a.data.time);controls=[];log=[]
        try:
            for t in range(cut,cut+145):
                q=a.joints.copy(); ref=a.data.ctrl[axes].copy()
                x=mpc.observed(e,t,camera.capture() if t%2==0 else None,q)
                g=actor.forecourt_command(t,e.eye,x,q,ref,home)
                u=np.clip((np.clip(g,lo,hi)-ref)/stride,-.8,.8)
                log.append(dict(tick=t,x=None if x is None else x.tolist(),q=q.tolist(),ref=ref.tolist(),
                    next_ref=np.clip(ref+stride*u,lo,hi).tolist(),bounce=actor.engine.bounced))
                controls.append(u.tolist());_,_,end,cap,_=a.exchange(u,mpc.FAR_COMMAND)
                if end or cap or a.rival_returns>=leg+2:
                    break
            shape=a.clearance_result();pp=a.point_placements()
            if shape['samples']!=round((a.data.time-began)/a.model.opt.timestep):
                raise ValueError('A physical suffix skipped a strict CAD sample')
            o=pp['delivered'][leg] if len(pp['delivered'])>leg else None
            landed=bool(o and o['near_landed'])
            states[mode]=dict(landed=landed,answered=landed and len(pp['delivered'])>leg+1,
                cadence=actor.evidence,controls=controls,observations=log,placements=pp,events=a.events,
                clearance=shape,offer=o)
        finally:
            a.sample_clearance(False)
    u=states['stock']['offer']
    matched=bool(u and all(v['offer'] and all(same_trace(u[k],v['offer'][k]) for k in ('far_contact','near_bounce'))
                           for v in states.values()))
    if all(v['offer'] for v in states.values()) and not matched:
        raise ValueError('Causal contrast changed incoming physical ball')
    if any(z['placements']['requested'][leg]['target']!=list(target) for z in states.values()):
        raise ValueError('Unexpected physical target')
    return dict(index=index,stage=stage,seed=seed,target=list(target),matched=matched,variants=states,
                offered_deep=None if not u or not u['receive_line'] else native.line(u['receive_line']),
                starting_q=q0.tolist(),status='done')


def run(out):
    row,live,ready,old=teacher(); p=_read(PLAN);packet=_read(PACKET);c=row['card']
    if p['packet_sha256']!=_hash(PACKET):
        raise ValueError('Changed scored teacher optical packet')
    result=_read(out) if out.exists() else dict(protocol_sha256=_hash(PLAN),rows=[],templates=packet['templates'],complete=False)
    if result['protocol_sha256']!=_hash(PLAN):
        raise ValueError('Changed goal roster')
    with np.load(mpc.BANK) as table:
        raw=mpc.DeepField(table,'local')
    with mpc.centered(ready):
        a=mpc.arena();cad=mpc.arena(detail='cad');check=CourtClearance(cad.model,articulated=True)
        res=mpc.resources(a);camera,rig,*_=res
        with patch.object(mpc,'Fragments',lambda b,e:Carried(b,e,practice=[course.PRACTICE],held={c['id']})):
            actor=Forecourt(mpc.BANK,raw,variant='elastic',budget=28,coverage=fresh.shot.Coverage(_read(fresh.RISK)))
        try:
            a.launch(live['shot'],card=c);a.partner.reset(seed=c['seed'])
            eye=mpc.Vision(rig,a.control_dt,raw=True);eye.eye=mpc.LearnedOptics(eye.eye,course.OPTIC)
            actor.begin();axes=a._arm_ids[1][2]
            for t,u in enumerate(live['actions'][:p['anchor']]):
                q=a.joints.copy();mpc.observed(eye,t,camera.capture() if t%2==0 and t>=p['anchor']-34 else None,q)
                actor.engine.observe(t,eye.eye,q,a.data.ctrl[axes].copy())
                a.exchange(u,mpc.FAR_COMMAND)
            if len(a.partner.requests)!=p['leg']:
                raise ValueError('Cannot redraw an opponent goal already requested')
            root=State(a);engine_root=body_snapshot(actor.engine)
            tests=[('seed',c['reply_seed'],p['seed_target'],False)]
            tests.extend(('graded',c['reply_seed'],np.asarray(p['seed_target'])+np.array(d),True)
                         for d in p['neighbor_deltas'])
            tests.extend(('natural_nearby',z['seed'],z['target'],False) for z in p['nearest_natural_streams'])
            tests.extend(('broad',s,STD(dict(c,reply_seed=s),p['leg'],'random',None)[0],False) for s in p['broad_streams'])
            for i,(stage,seed,target,changed) in enumerate(tests):
                if i<len(result['rows']):continue
                if stage!='seed' and not result['rows'][0]['variants'].get('learned',{}).get('answered'):
                    result['status']='rejected_exact_optical_phase';break
                if stage=='broad' and sum(r['variants'].get('learned',{}).get('answered',False) for r in result['rows'] if r['stage']=='natural_nearby')<2:
                    result['status']='insufficient_ordinary_nearby_transfer';break
                def assigned(card,leg,protocol,rng):
                    if changed and leg==p['leg']:
                        return tuple(target),'tt54_declared_local_practice'
                    return STD(card,leg,protocol,rng)
                lib=optical.RGBTaught(result['templates'])
                with patch.object(opponent,'draw',assigned):
                    got=one(a,res,actor,root,eye,engine_root,c,live,p,lib,seed,target,stage,i,check)
                v=got['variants'].get('learned',{})
                if (v.get('answered') and not v['clearance']['penetration_samples'] and got['matched']):
                    witness=dict(shot=live['shot'],actions=live['actions'][:p['cut']]+v['controls'],events=v['events'],placements=v['placements'])
                    with patch.object(opponent,'draw',assigned):
                        v['full_cad']=native.full_verify(ready,dict(c,reply_seed=c['reply_seed']),witness,check) if changed else independent_verify(
                                                   ready,c,seed,witness,check,p['anchor'])
                    if v['full_cad']['penetration_samples']==0 and stage=='graded':
                        template=optical.extract(v['observations'],BladeIK(a),f'own-central-{i}',p['cut'])
                        if template is not None:
                            template['p']=cue_prototype(v['observations'],template)
                            result['templates'].append(optical.plain(template))
                            v['acquired']=True
                result['rows'].append(got)
                optical.save(out,result)
                print(stage,i,got['offered_deep'] if got['status']=='done' else got['status'],
                    [(k,z['landed'],z['answered'],len(z['cadence']),sum(t['issued'] for t in z['cadence'])) for k,z in got['variants'].items()],flush=True)
        finally:
            actor.close();camera.close();a.close();cad.close()
    result.update(complete=True,status=result.get('status','finished_physical_course'))
    optical.save(out,result)


def independent_verify(ready,card,seed,witness,check,anchor):
    with mpc.centered(ready):
        a=mpc.arena(width=64,height=48)
        try:
            a.launch(witness['shot'],card=card);a.partner.reset(seed=card['seed']);guarded(a,check)
            for t,u in enumerate(witness['actions']):
                if t==anchor:
                    a.card=dict(card,reply_seed=seed)
                a.exchange(u,mpc.FAR_COMMAND)
            if digest_json(a.events)!=digest_json(witness['events']) or digest_json(a.point_placements())!=digest_json(witness['placements']):
                raise ValueError('Independent physical replay diverged')
            clearance=a.clearance_result()
            if clearance['samples']!=round(a.data.time/a.model.opt.timestep):
                raise ValueError('Independent CAD skipped a physics state')
            return clearance
        finally:
            a.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('capture','plan','run'))
    parser.add_argument('--out',type=Path,default=RESULT)
    args=parser.parse_args()
    if args.mode=='capture':capture()
    elif args.mode=='plan':plan()
    else:run(args.out)
