"""Existing five TT50 neutral draws, actual early fork, official raw camera base.

The lead wing controller supplies an install(raw,checker,reference,enabled=,
frames=) factory. This evaluates its early control without repeating full games.
"""
import argparse
from importlib import import_module
from pathlib import Path
from time import monotonic,process_time
from unittest.mock import patch
import numpy as np
from . import tt52_stock_repertory as rp
from .tt51_low_reverse_curriculum import plain
from .tt37_opponent_audit import CourtClearance,guarded


def restore_readiness_from_actual_images(player,rig,dt,readings):
    ready=getattr(player,'ready',None)
    if ready is None:return
    ready.reset()
    eye=rp.mpc.Vision(rig,dt,raw=True)
    eye.eye=rp.mpc.LearnedOptics(eye.eye,rp.pf.prior.course.OPTIC)
    for tick,frames,q,ref in readings:
        x=rp.mpc.observed(eye,tick,frames,q)
        ready.observe(tick,x,q,ref,frames)


def run(output,module_name,n=5):
    if output.exists():raise ValueError('Use a new result file')
    module=import_module(module_name);factory=module.install
    prior=rp.pf.prior;mpc=rp.mpc;src=rp._read(prior.ARCHIVE);case,original=prior.terminal(src)
    reg=rp._read(prior.PLAN);roster=reg['rows'][1:];root=reg['anchor'];keep=reg['keep_old_commands_through']
    if len(roster)!=5 or any(not r['split'].startswith('disjoint-') for r in roster):
        raise ValueError('Existing five neutral cases changed')
    if n not in (2,3,5):raise ValueError('Only the complete cohort or registered first-two/three diagnostics are supported')
    roster=roster[:n]
    doc=dict(schema='tt51_earlier_wing_neutrals_v1',source_sha256=rp._hash(Path(__file__)),
        integration_module=module_name,integration_sha256=rp._hash(Path(module.__file__)),
        protocol_sha256=rp._hash(prior.PLAN),carry_sha256=rp._hash(prior.season.CARRY),rows=[],complete=False,
        roster_count=n,subset_role='Full existing cohort' if n==5 else f'Known-label first-{n} mechanism diagnostic; not held evidence',
        scope='Existing five TT50 neutral draws, actual source arm at3999 with real last32 camera frames and all past encoders. Original commands through4003. Normal broad far draws and official raw.get base. Earlier controlled fork, not independent full games.',
        seconds=450,max_actual_control_ticks=10000)
    clock=monotonic();cpu=process_time();deadline=clock+450;paid=0
    def save(status=None):
        doc.update(actual_control_ticks=paid,wall_seconds=monotonic()-clock,cpu_seconds=process_time()-cpu)
        if status:doc.update(complete=True,status=status,
            integration_frozen=doc['integration_sha256']==rp._hash(Path(module.__file__)),
            evaluator_frozen=doc['source_sha256']==rp._hash(Path(__file__)))
        rp._write(output,plain(doc))
    with np.load(mpc.BANK) as z:raw=mpc.DeepField(z,'local')
    with mpc.centered(src['ready']):
        a=mpc.arena();cad=mpc.arena(detail='cad');clear=CourtClearance(cad.model,articulated=True)
        camera,rig,stride,lo,hi=mpc.resources(a);tap=rp.old.Tap(camera);axes=a._arm_ids[1][2]
        checker=prior.Reach(cad,src['ready'],src['ready']);models={}
        try:
            for mode in ('stock','cold','warm'):
                with patch.object(mpc,'Fragments',lambda f,e:prior.Carried(f,e,practice=[prior.course.PRACTICE],held={case['id']})):
                    if mode=='stock':
                        model=prior.season.Player(mpc.BANK,raw,variant=src['variant'],budget=src['budget'],
                            coverage=prior.fresh.shot.Coverage(rp._read(prior.fresh.RISK)))
                    else:
                        public=dict(ready=src['ready'],variant=src['variant'],budget=src['budget'])
                        model=factory(raw,checker,public,enabled=mode=='warm',frames=tap)
                model.tap=tap
                carry=rp._read(prior.season.CARRY)
                prior.course.resume(model,prior.season.CARRY.with_suffix('').with_suffix('.npz'),carry)
                models[mode]=model
            history=prior.prefix(a,case['card'],original,root,camera,axes,window=32);paid+=root
            anchor=prior.State(a)
            for item in roster:
                row=dict(index=len(doc['rows']),teacher=0,cut=root,delta=[0,0,0,0],goal=item['goal'],
                    random_reply_stream=item['stream'],stage='earlier_natural_neutral',split=item['split'],variants={})
                for mode,player in models.items():
                    if monotonic()>=deadline or paid>=10000:save('bounded_partial');return
                    anchor.restore(a);a.card=dict(case['card'],reply_seed=item['stream'])
                    observer=prior.Observer(a,rig,player,history);eye=observer.restore(player)
                    restore_readiness_from_actual_images(player,rig,a.control_dt,history)
                    guarded(a,clear);initial=player.priors.digest();home=np.asarray(src['ready'])
                    controls=[];observations=[];sensed=False;completed=False
                    for tick in range(root,root+240):
                        if monotonic()>=deadline or paid>=10000:break
                        q=a.joints.copy();ref=a.data.ctrl[axes].copy();frames=tap.capture() if tick%2==0 else None
                        x=mpc.observed(eye,tick,frames,q)
                        if tick<=keep:
                            if hasattr(player,'observe_only'):player.observe_only(tick,frames,eye.eye,x,q,ref)
                            else:player.engine.observe(tick,eye.eye,q,ref)
                            u=np.asarray(original['actions'][tick])
                        else:
                            if x is not None and x[4]<-.9 and x[1]<-.9:sensed=True
                            base=home
                            if x is not None and (x[4]<-.9 or (sensed and x[1]<.50)):
                                target,confidence=raw.get(x,q)
                                if target is not None:base=home+confidence*(target-home)
                            g=player.step(tick,eye.eye,x,q,ref,base)
                            u=np.clip((np.clip(g,lo,hi)-ref)/stride,-.8,.8)
                        observations.append(dict(tick=tick,x=None if x is None else x.tolist(),q=q.tolist(),ref=ref.tolist(),
                            next_ref=np.clip(ref+stride*u,lo,hi).tolist()))
                        controls.append(u.tolist());_,_,done,capped,_=a.exchange(u,mpc.FAR_COMMAND);paid+=1
                        if done or capped or a.rival_returns>=44:completed=True;break
                    placements=a.point_placements();offers=placements['delivered']
                    offer=offers[case['leg']] if len(offers)>case['leg'] else None
                    if placements['requested'][case['leg']]['target']!=list(rp.draw(a.card,case['leg'],'random',None)[0]):
                        raise RuntimeError('Opponent draw changed')
                    if controls[:keep-root+1]!=original['actions'][root:keep+1]:raise RuntimeError('Changed protected earlier controls')
                    if player.priors.digest()!=initial:raise RuntimeError('Frozen ordinary memory changed')
                    result=dict(legal=bool(offer and offer['near_landed']),answered=a.rival_returns>=44,completed=completed,
                        actual_offer=offer,controls=controls,observations=observations,clearance=a.clearance_result(),
                        actual_control_ticks=len(controls),event_sha256=prior.digest_json(a.events),
                        opts=getattr(player,'repertory_log',[]),body_trace=getattr(player,'body_trace',[]))
                    row['variants'][mode]=result;doc['in_progress']=row;save()
                values=list(row['variants'].values())
                if all(v['actual_offer'] is not None for v in values):
                    row['matched_incoming']=all(all(prior.same_trace(v['actual_offer'][k],values[0]['actual_offer'][k])
                        for k in ('far_contact','near_bounce')) for v in values[1:])
                    if not row['matched_incoming']:raise RuntimeError('Different actual incoming between controllers')
                doc['rows'].append(row);doc.pop('in_progress',None);save()
                print(item['stream'],[(m,v['legal'],v['answered']) for m,v in row['variants'].items()],flush=True)
            save('completed_existing_neutral_roster')
        finally:
            for model in models.values():model.close()
            camera.close();a.close();cad.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--module',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=int,choices=(2,3,5),default=5)
    args=p.parse_args();run(args.out,args.module,args.n)
