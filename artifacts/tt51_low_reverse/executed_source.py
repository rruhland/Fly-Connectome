"""Bounded RGB/reward-only reverse curriculum around two physically earned low strokes.

Actual command prefixes provide the disclosed practice starting states. New feeds
come from physical far-paddle goals, never ball teleports. No native ball/contact
metadata is used to extract or query the learned templates.
"""
import argparse
from copy import deepcopy
from pathlib import Path
from time import monotonic,process_time
from unittest.mock import patch
import mujoco
import numpy as np

from . import tt50_conditional_prep as arm
from . import tt37_opponent as opponent
from .tt50_optical_contact_school import inputs
from .tt37_opponent_audit import CourtClearance,guarded
from .tt39_opportunity_map import projected_height

ROOT=Path('artifacts/tt51_low_reverse')


def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [plain(x) for x in v]
    return v


def extract(log,ik,name,start):
    inbound=[r for r in log if r['tick']>=start and r['x'] is not None and r['x'][4]<-.9]
    if not inbound:return None
    outbound=next((r['tick'] for r in log if r['tick']>inbound[0]['tick']
                   and r['x'] is not None and r['x'][4]>.9),None)
    if outbound is not None:inbound=[r for r in inbound if r['tick']<outbound]
    if not inbound:return None
    geometry=ik.model.geom('learner_racket').size
    radius=float(geometry[0]);half=float(geometry[1])
    def gap(r):
        center,normal=ik.pose(np.asarray(r['q'])[:5]);delta=np.asarray(r['x'])[:3]-center
        axial=abs(float(delta@normal));tangent=np.sqrt(max(0.,float(delta@delta)-axial**2))
        return np.hypot(max(0.,axial-half),max(0.,tangent-radius))-.02
    cue=min(inbound,key=gap);touch=cue['tick']
    crossing=min((r for r in inbound if r['tick']<=touch),key=lambda r:abs(r['x'][1]+1.10))
    x=np.asarray(crossing['x']);prototype=np.r_[x[0],x[2],x[3:6]]
    relevant=[r for r in log if touch-23<=r['tick']<=touch+10]
    if len(relevant)<6:return None
    return dict(name=name,p=prototype,relative=np.array([r['tick']-touch for r in relevant]),
        refs=np.array([r['next_ref'] for r in relevant]),depth=float(cue['x'][1]),
        source_phase_tick=touch,minimum_camera_body_gap=float(gap(cue)),
        extraction='RGB incoming/context; observed ball versus nominal FK from own encoders; actual issued next references; scalar legal reward authorizes storage')


class RGBTaught(arm.Taught):
    def __init__(self,rows):
        self.rows=[deepcopy(r) for r in rows]


class NoFarPreview:
    """The reverse practice studies body/late skill, not the parallel far-RGB cue."""
    def reset(self):pass
    def update(self,*args):return None


def seeded_rows(packet,ik):
    rows=[]
    for sample in packet['rows']:
        if sample['reward']!='physical legal and answered':continue
        row=extract(sample['observations'],ik,sample['name'],sample['mode_start'])
        if row is not None:rows.append(row)
    if len(rows)!=2:raise ValueError('Both scored source templates need legal RGB/body extraction')
    return rows


def save(path,value):arm._write(path,plain(value))


def run(args):
    assert not args.out.exists();ROOT.mkdir(parents=True,exist_ok=True)
    src=arm._read(arm.ARCHIVE);case,archive=arm.terminal(src);port,teachers=inputs()
    protocol=arm._read(arm.PLAN);packet=arm._read(arm.SKILL);policy=arm._read(arm.DIRECT_MODEL)
    local=arm.Local(policy['rows'],[z['context'] for z in policy['easy']])
    motor=arm.ContinuousMap(arm._read(arm.TRAIN))
    doc=dict(schema='tt51_rgb_low_reverse_v1',source_sha256=arm._hash(Path(__file__)),
        source_teacher_sha256=arm._hash(arm.SKILL),verified_school_sha256=arm._hash(arm.ROOT/'contact-school.json.gz'),
        practice_scope='Actual teacher command prefix to a declared handoff. Naturally changed far goal, shared cold/taught physical handoff, normal RGB/q thereafter. Not official full-point qualification.',
        input_scope='No native_touch or x_at_receive_line in template extraction or policy. Native incoming/score/CAD remain evaluator-only.',
        seconds=900,max_actual_control_ticks=12000,rows=[],complete=False,
        template_extraction='closest observed ball-to-paddle boundary using RGB and own encoder FK; RGB incoming crossing; actual issued references',
        gate='Both exact seeds must score; then≥3/8 new goal-neighbor growing returns before earlier-state expansion.',
        modes=['cold','rgb-seed','growing'])
    dependencies=[Path(arm.__file__),Path(__file__).with_name('tt50_optical_contact_school.py'),
        Path(__file__).with_name('tt44_paired_credit.py'),Path(__file__).with_name('tt37_optical_clinic.py')]
    doc['dependency_hashes']={str(p):arm._hash(p) for p in dependencies}
    doc['held_reply_streams']=np.random.default_rng(510257).integers(1,2**30,4).tolist()
    clock=monotonic();cpu=process_time();deadline=clock+900;paid=0
    with np.load(arm.mpc.BANK) as z:raw=arm.mpc.DeepField(z,'local')
    with arm.mpc.centered(src['ready']):
        a=arm.mpc.arena();cad=arm.mpc.arena(detail='cad');clear=CourtClearance(cad.model,articulated=True)
        camera,rig,stride,lo,hi=arm.mpc.resources(a);axes=a._arm_ids[1][2]
        checker=arm.Reach(cad,src['ready'],src['ready'])
        with patch.object(arm.mpc,'Fragments',lambda f,e:arm.Carried(f,e,practice=[arm.course.PRACTICE],held={case['id']})):
            actor=arm.Arm(arm.mpc.BANK,raw,readout=local,examples=(),
                coverage=arm.fresh.shot.Coverage(arm._read(arm.fresh.RISK)),budget=src['budget'],variant=src['variant'],
                predictor_path=arm.CAMERA,affine=motor,taught=None,checker=checker,
                waypoint=port['waypoint'],approach=port['approach'])
        actor.predictor=NoFarPreview()
        try:
            past={**src,'rows':[None]*len(arm._read(arm.season.CARRY)['rows'])+src['rows']}
            arm.course.resume(actor,arm.BANK,past)
            seed=seeded_rows(packet,checker.ik);growing=deepcopy(seed)
            doc['initial_templates']=plain(seed)
            root=protocol['anchor']
            history=arm.prefix(a,case['card'],archive,root,camera,axes,window=32);paid+=root
            saved=arm.State(a);cold=arm.Observer(a,rig,actor,history)
            doc['paid_full_prefix_ticks']=root

            def finish(status):
                doc.update(status=status,complete=True,actual_control_ticks=paid,
                    wall_seconds=monotonic()-clock,cpu_seconds=process_time()-cpu,
                    final_templates=plain(growing),source_frozen=doc['source_sha256']==arm._hash(Path(__file__)),
                    dependencies_frozen=all(arm._hash(p)==h for p,h in doc['dependency_hashes'].items()))
                save(args.out,doc)

            def attempt(ti,delta,earlier,stage,index,random_stream=None):
                nonlocal paid
                t=teachers[ti];stream=t.get('stream',case['card']['reply_seed'])
                if random_stream is not None:stream=random_stream
                source=t['tape']['placements']['requested'][case['leg']]
                goal=np.asarray(source['target'])+np.asarray(delta)
                cut=t['cut']-earlier
                saved.restore(a);a.card=dict(case['card'],reply_seed=stream)
                eye=cold.restore(actor);outgoing=None;inbound=None
                tape=t['tape'];original_draw=arm.draw
                def draw(card,leg,protocol,rng):
                    if leg==case['leg'] and random_stream is None:
                        return tuple(goal),source['draw'] if not np.any(delta) else 'tt51_reverse_goal_practice'
                    return original_draw(card,leg,protocol,rng)
                with patch.object(opponent,'draw',draw):
                    for tick in range(root,cut):
                        if monotonic()>=deadline or paid>=12000:return None
                        q=a.joints.copy();ref=a.data.ctrl[axes].copy();frame=camera.capture() if tick%2==0 else None
                        x=arm.mpc.observed(eye,tick,frame,q);actor.engine.observe(tick,eye.eye,q,ref)
                        if x is not None and x[4]>.9:outgoing=tick
                        if outgoing is not None and x is not None and x[4]<-.9 and inbound is None and tick-outgoing<40:inbound=tick
                        _,_,done,capped,_=a.exchange(tape['actions'][tick],arm.mpc.FAR_COMMAND);paid+=1
                        if done or capped:
                            ended=dict(stage=stage,index=index,teacher=ti,status='ended_before_handoff',
                                actual_control_ticks=tick-root+1,variants={})
                            doc['rows'].append(ended);doc['actual_control_ticks']=paid;save(args.out,doc)
                            return ended
                    state=arm.State(a);eye0=deepcopy(eye);e=actor.engine
                    body=mujoco.MjData(e.body_model);mujoco.mj_copyData(body,e.body_model,e.body_state)
                    keys=('previous','rate','tick','reference','bounced')
                    engine0={k:deepcopy(getattr(e,k)) for k in keys}
                    before=arm.digest_json(a.events);q0=a.joints.copy();ref0=a.data.ctrl[axes].copy()
                    row=dict(stage=stage,index=index,teacher=ti,goal=goal.tolist(),delta=list(delta),cut=cut,
                        release_ticks_earlier=earlier,starting_q=q0.tolist(),starting_ref=ref0.tolist(),prefix_event_sha256=before,
                        prefix_new_control_ticks=cut-root,random_reply_stream=random_stream,variants={})
                    for mode in ('cold','rgb-seed','growing'):
                        if monotonic()>=deadline or paid>=12000:break
                        state.restore(a);guarded(a,clear);actor.begin()
                        actor.policy_mode='stock' if mode=='cold' else 'body-learned'
                        actor.taught=RGBTaught(seed if mode=='rgb-seed' else growing) if mode!='cold' else None
                        actor.y_out=outgoing;actor.seen_inbound_at=inbound
                        fixed_prior=actor.priors.digest()
                        eye=deepcopy(eye0);mujoco.mj_copyData(e.body_state,e.body_model,body)
                        for k,v in engine0.items():setattr(e,k,deepcopy(v))
                        e.tracker=eye.eye;e.latest=eye.eye.latest
                        if mode!='cold':
                            actor.hard_active=True;actor.plan_t=cut;actor.plan=np.repeat(ref0[None,:],20,axis=0)
                        controls=[];observations=[];completed=False
                        for tick in range(cut,cut+140):
                            if monotonic()>=deadline or paid>=12000:break
                            q=a.joints.copy();ref=a.data.ctrl[axes].copy();frame=camera.capture() if tick%2==0 else None
                            x=arm.mpc.observed(eye,tick,frame,q)
                            goal_ref=actor.blended(tick,frame,eye.eye,x,q,ref,actor.priors.home)
                            u=np.clip((np.clip(goal_ref,lo,hi)-ref)/stride,-.8,.8)
                            observations.append(dict(tick=tick,x=None if x is None else x.tolist(),q=q.tolist(),
                                ref=ref.tolist(),next_ref=np.clip(ref+stride*u,lo,hi).tolist()))
                            controls.append(u.tolist());_,_,done,capped,_=a.exchange(u,arm.mpc.FAR_COMMAND);paid+=1
                            if done or capped or a.rival_returns>=44:completed=True;break
                        delivery=a.point_placements()['delivered']
                        offer=delivery[case['leg']] if len(delivery)>case['leg'] else None
                        legal=bool(offer and offer['near_landed']);answered=bool(a.rival_returns>=44)
                        shape=a.clearance_result()
                        if actor.priors.digest()!=fixed_prior:raise RuntimeError('Frozen ordinary motor bank changed')
                        outcome=dict(legal=legal,answered=answered,completed=completed,controls=controls,observations=observations,
                            actual_offer=offer,trials=actor.shots,clearance=shape,actual_control_ticks=len(controls),
                            event_sha256=arm.digest_json(a.events),templates_used=len(actor.taught.rows) if actor.taught else 0)
                        row['variants'][mode]=outcome
                        if mode=='growing' and stage!='seed' and legal and answered and not shape['penetration_samples']:
                            learned=extract(observations,checker.ik,f'own-reward-{stage}-{index}',cut)
                            if learned is not None:growing.append(learned);outcome['new_rgb_template_added']=True
                        doc.update(in_progress=row,actual_control_ticks=paid,wall_seconds=monotonic()-clock,
                            cpu_seconds=process_time()-cpu)
                        save(args.out,doc)
                    variants=list(row['variants'].values())
                    if len(variants)==3 and all(v['actual_offer'] is not None for v in variants):
                        row['matched_incoming']=all(all(arm.same_trace(v['actual_offer'][k],variants[0]['actual_offer'][k])
                            for k in ('far_contact','near_bounce')) for v in variants[1:])
                        if not row['matched_incoming']:raise RuntimeError('Paired policies did not receive the same physical ball')
                    doc['rows'].append(row)
                    doc.pop('in_progress',None)
                    doc.update(actual_control_ticks=paid,wall_seconds=monotonic()-clock,cpu_seconds=process_time()-cpu)
                    save(args.out,doc)
                    print(stage,index,ti,earlier,[(m,v['legal'],v['answered'],len(v['trials'])) for m,v in row['variants'].items()],flush=True)
                    return row

            for ti in range(2):
                r=attempt(ti,[0,0,0,0],0,'seed',ti)
                if r is None:finish('bounded_seed_partial');return
            seeds=doc['rows'][-2:]
            if not all(r.get('variants',{}).get('rgb-seed',{}).get('legal',False) for r in seeds):
                finish('rejected_rgb_only_seed_did_not_reproduce_both_teachers');return
            if args.seed_only:finish('rgb_only_seed_verified');return
            deltas=([.012,0,0,0],[-.012,0,0,0],[0,0,.008,0],[0,0,-.008,0])
            for ti in range(2):
                for delta in deltas:
                    if attempt(ti,delta,0,'neighbors',len(doc['rows'])) is None:
                        finish('bounded_neighbor_partial');return
            trained=[r for r in doc['rows'] if r['stage']=='neighbors']
            wins=sum(r.get('variants',{}).get('growing',{}).get('legal',False) for r in trained)
            if wins<3:finish('rejected_cannot_expand_teacher_neighborhood');return
            for ti in range(2):
                for delta in ([.025,0,.012,.06],[-.025,0,-.012,-.06]):
                    if attempt(ti,delta,6,'earlier_held',len(doc['rows'])) is None:
                        finish('bounded_earlier_partial');return
            earlier=[r for r in doc['rows'] if r['stage']=='earlier_held']
            if sum(r.get('variants',{}).get('growing',{}).get('legal',False) for r in earlier)<2:
                finish('rejected_earlier_state_expansion');return
            for i,stream in enumerate(doc['held_reply_streams']):
                if attempt(i%2,[0,0,0,0],6,'untouched_broad_late_state',len(doc['rows']),stream) is None:
                    finish('bounded_broad_late_state_partial');return
            finish('completed_reverse_curriculum_diagnostic_no_official_promotion')
        finally:actor.close();camera.close();a.close();cad.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'course.json.gz')
    p.add_argument('--seed-only',action='store_true');run(p.parse_args())
