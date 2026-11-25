"""Reward-gated reverse body/center curriculum through the unchanged stock search."""
import argparse
from copy import deepcopy
from pathlib import Path
from time import monotonic,process_time
from unittest.mock import patch
import mujoco
import numpy as np
from . import tt52_stock_repertory as rp
from . import tt37_opponent as opponent
from .tt51_low_reverse_curriculum import extract,plain,RGBTaught
from .tt51_low_body_bridge import BodyContinuation,earlier_by_body
from .tt50_optical_contact_school import inputs
from .tt37_opponent_audit import CourtClearance,guarded


def run(output):
    if output.exists():raise ValueError('New physical course requires a new output')
    prior=rp.pf.prior;mpc=rp.mpc;src=rp._read(prior.ARCHIVE);case,original=prior.terminal(src)
    _,teachers=inputs();packet=rp._read(prior.SKILL);root=rp._read(prior.PLAN)['anchor']
    seed=rp._read(rp.low_artifact)['initial_templates'];growing=deepcopy(seed)
    modes=('cold','late-only','body-seed','body-growing')
    doc=dict(schema='tt51_low_stock_bridge_v1',source_sha256=rp._hash(Path(__file__)),rows=[],complete=False,
        modes=modes,initial_templates=seed,body_source_sha256=rp._hash(Path(BodyContinuation.__module__.replace('.','/')+'.py')),
        repertory_source_sha256=rp._hash(Path(rp.__file__)),
        scope='Declared physically replayed earlier low-practice states. Encoder-phase continuation from two rewarded trajectories. Same ordinary stock search and RGB window in all arms; no source tick/id, private ball or incoming goal supplied to policy. Not official broad normal-start games.',
        gate='Four .5/1.2rad earlier tests; >=2 body-seed legal returns and >=1 gain over late-only, then translate first teacher goal x by+.024,+.04,+.06,+.08,+.10,+.12 with z/pace fixed; stop after2 successive body-growing misses.',
        seconds=900,max_actual_control_ticks=12000)
    dependencies=[Path(rp.__file__),Path(rp.old.__file__),Path(rp.pf.__file__),Path(prior.__file__),
        Path(BodyContinuation.__module__.replace('.','/')+'.py')]
    doc['dependency_hashes']={str(p):rp._hash(p) for p in dependencies}
    doc['carry_sha256']=rp._hash(prior.season.CARRY)
    clock=monotonic();cpu=process_time();paid=0;deadline=clock+900
    def save(status=None):
        doc.update(actual_control_ticks=paid,wall_seconds=monotonic()-clock,cpu_seconds=process_time()-cpu,
            final_templates=plain(growing))
        if status:doc.update(complete=True,status=status,
            source_frozen=doc['source_sha256']==rp._hash(Path(__file__)),
            dependencies_frozen=all(rp._hash(Path(p))==h for p,h in doc['dependency_hashes'].items()))
        rp._write(output,plain(doc))
    with np.load(mpc.BANK) as z:raw=mpc.DeepField(z,'local')
    with mpc.centered(src['ready']):
        a=mpc.arena();cad=mpc.arena(detail='cad');clear=CourtClearance(cad.model,articulated=True)
        camera,rig,stride,lo,hi=mpc.resources(a);axes=a._arm_ids[1][2]
        from .tt29_rally_physics import BladeIK
        ik=BladeIK(cad)
        with patch.object(mpc,'Fragments',lambda f,e:prior.Carried(f,e,practice=[prior.course.PRACTICE],held={case['id']})):
            player=rp.Arm(mpc.BANK,raw,mode='stock',finish=rp.pf.Finish(),variant=src['variant'],
                budget=src['budget'],coverage=prior.fresh.shot.Coverage(rp._read(prior.fresh.RISK)))
        try:
            carry=rp._read(prior.season.CARRY)
            carry_bank=prior.season.CARRY.with_suffix('').with_suffix('.npz')
            prior.course.resume(player,carry_bank,carry)
            history=prior.prefix(a,case['card'],original,root,camera,axes,window=32);paid+=root
            anchor=prior.State(a);warm=prior.Observer(a,rig,player,history)
            body=BodyContinuation(packet,[t['cut'] for t in teachers])
            def attempt(ti,radius,dx,stage):
                nonlocal paid
                t=teachers[ti];earlier=earlier_by_body(packet['rows'][ti],t['cut'],radius);cut=t['cut']-earlier
                request=t['tape']['placements']['requested'][case['leg']]
                goal=np.asarray(request['target'])+np.array([dx,0.,0.,0.])
                anchor.restore(a);a.card=dict(case['card'],reply_seed=t.get('stream',case['card']['reply_seed']))
                eye=warm.restore(player)
                def draw(card,leg,protocol,rng):
                    if leg==case['leg']:return tuple(goal),request['draw'] if dx==0 else 'tt51_low_center_practice'
                    return rp.draw(card,leg,protocol,rng)
                with patch.object(opponent,'draw',draw):
                    for tick in range(root,cut):
                        if monotonic()>=deadline or paid>=12000:return None
                        q=a.joints.copy();ref=a.data.ctrl[axes].copy();frames=camera.capture() if tick%2==0 else None
                        x=mpc.observed(eye,tick,frames,q);player.observe_only(tick,frames,eye.eye,x,q,ref)
                        _,_,done,capped,_=a.exchange(t['tape']['actions'][tick],mpc.FAR_COMMAND);paid+=1
                        if done or capped:return None
                    snap=prior.State(a);screen=deepcopy(eye);e=player.engine
                    native=mujoco.MjData(e.body_model);mujoco.mj_copyData(native,e.body_model,e.body_state)
                    values={k:deepcopy(getattr(e,k)) for k in ('previous','rate','tick','reference','bounced')}
                    q0=a.joints.copy();ref0=a.data.ctrl[axes].copy()
                    qteacher=np.asarray(next(r['q'] for r in packet['rows'][ti]['observations'] if r['tick']==t['cut']))
                    row=dict(index=len(doc['rows']),stage=stage,teacher=ti,cut=cut,goal=goal.tolist(),delta=[dx,0,0,0],
                        random_reply_stream=None,prefix_new_control_ticks=cut-root,starting_q=q0.tolist(),starting_ref=ref0.tolist(),
                        max_joint_distance_from_teacher_handoff=float(np.max(abs(q0-qteacher))),
                        max_joint_distance_from_normal_ready=float(np.max(abs(q0-src['ready']))),variants={})
                    for mode in modes:
                        if monotonic()>=deadline or paid>=12000:break
                        snap.restore(a);guarded(a,clear);player.begin();body.reset()
                        player.repertory=None if mode=='cold' else rp.Joined(None,RGBTaught(growing if mode=='body-growing' else seed))
                        initial=player.priors.digest();eye=deepcopy(screen)
                        mujoco.mj_copyData(e.body_state,e.body_model,native)
                        for k,v in values.items():setattr(e,k,deepcopy(v))
                        e.tracker=eye.eye;e.latest=eye.eye.latest
                        opened=False;hold=ref0.copy();controls=[];observations=[];completed=False
                        for tick in range(cut,cut+160):
                            if monotonic()>=deadline or paid>=12000:break
                            q=a.joints.copy();ref=a.data.ctrl[axes].copy();frames=camera.capture() if tick%2==0 else None
                            x=mpc.observed(eye,tick,frames,q)
                            moving=mode.startswith('body') and not body.finished
                            if moving:
                                proposed=body.command(q,ref)
                                if proposed is not None:hold=proposed
                            eligible=x is not None and x[4]<-.9 and -1.62<x[1]<-.49
                            opened=opened or (eligible and not (mode.startswith('body') and not body.finished))
                            if opened:g=player.choose(tick,frames,eye.eye,x,q,ref,player.priors.home)
                            else:player.observe_only(tick,frames,eye.eye,x,q,ref);g=hold
                            u=np.clip((np.clip(g,lo,hi)-ref)/stride,-.8,.8)
                            observations.append(dict(tick=tick,x=None if x is None else x.tolist(),q=q.tolist(),ref=ref.tolist(),
                                next_ref=np.clip(ref+stride*u,lo,hi).tolist()))
                            controls.append(u.tolist());_,_,done,capped,_=a.exchange(u,mpc.FAR_COMMAND);paid+=1
                            if done or capped or a.rival_returns>=44:completed=True;break
                        offers=a.point_placements()['delivered'];offer=offers[case['leg']] if len(offers)>case['leg'] else None
                        legal=bool(offer and offer['near_landed']);answered=a.rival_returns>=44
                        result=dict(legal=legal,answered=answered,completed=completed,actual_offer=offer,controls=controls,
                            observations=observations,actual_control_ticks=len(controls),opts=player.repertory_log,
                            body_continuation=deepcopy(body.records) if mode.startswith('body') else [],
                            clearance=a.clearance_result(),event_sha256=prior.digest_json(a.events),templates_used=len(growing) if mode=='body-growing' else len(seed) if mode!='cold' else 0)
                        if player.priors.digest()!=initial:raise RuntimeError('Frozen stock changed')
                        if (mode=='body-growing' and stage=='center_translation' and legal and answered
                                and not result['clearance']['penetration_samples']):
                            learned=extract(observations,ik,f"paid-center-{row['index']}",cut)
                            if learned is not None:growing.append(learned);result['new_template_added']=True
                        row['variants'][mode]=result;doc['in_progress']=row;save()
                    values=list(row['variants'].values())
                    if len(values)==4 and all(v['actual_offer'] is not None for v in values):
                        row['matched_incoming']=all(all(prior.same_trace(v['actual_offer'][k],values[0]['actual_offer'][k])
                            for k in ('far_contact','near_bounce')) for v in values[1:])
                        if not row['matched_incoming']:raise RuntimeError('Unequal actual paired incoming')
                    doc['rows'].append(row);doc.pop('in_progress',None);save()
                    print(stage,ti,radius,dx,[(m,v['legal'],v['answered'],sum(o['accepted'] for o in v['opts']))
                        for m,v in row['variants'].items()],flush=True)
                    return row
            for ti in range(2):
                for radius in (.5,1.2):
                    if attempt(ti,radius,0.,'earlier_body') is None:save('bounded_body_partial');return
            gains=sum(r['variants']['body-seed']['legal'] and not r['variants']['late-only']['legal'] for r in doc['rows'])
            if sum(r['variants']['body-seed']['legal'] for r in doc['rows'])<2 or gains<1:
                save('rejected_no_causal_body_gain');return
            misses=0
            for dx in (.024,.04,.06,.08,.10,.12):
                r=attempt(0,1.2,dx,'center_translation')
                if r is None:save('bounded_center_partial');return
                misses=0 if r['variants']['body-growing']['legal'] else misses+1
                if misses>=2:save('stopped_center_curriculum_two_consecutive_misses');return
            save('completed_stock_bridge_practice_no_official_promotion')
        finally:player.close();camera.close();a.close();cad.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('artifacts/tt51_low_reverse/stock-bridge.json.gz'))
    run(p.parse_args().out)
