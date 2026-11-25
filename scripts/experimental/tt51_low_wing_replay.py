"""Fresh-controller, fresh-risk and full-CAD verification of an earned wing receipt."""
import argparse
from importlib import import_module
from pathlib import Path
from time import monotonic,process_time
from unittest.mock import patch
import numpy as np
from . import tt52_stock_repertory as rp
from .tt51_low_early_neutrals import restore_readiness_from_actual_images
from .tt51_low_reverse_curriculum import plain
from .tt37_opponent_audit import CourtClearance,guarded,verify_clearance


def run(source,output,index):
    d=rp._read(source);row=d['rows'][index];module=import_module(d['integration_module'])
    if d['integration_sha256']!=rp._hash(Path(module.__file__)):raise ValueError('Integration source changed')
    prior=rp.pf.prior;mpc=rp.mpc;src=rp._read(prior.ARCHIVE);case,original=prior.terminal(src)
    reg=rp._read(prior.PLAN);root=reg['anchor'];keep=reg['keep_old_commands_through']
    doc=dict(schema='tt51_fresh_wing_receipt_verification_v1',source_sha256=rp._hash(source),index=index,
        verifier_sha256=rp._hash(Path(__file__)),rows=[],complete=False,
        integration_sha256=d['integration_sha256'],risk_sha256=rp._hash(rp.risk_model),
        carry_sha256=rp._hash(prior.season.CARRY),bank_sha256=rp._hash(module.BANK))
    clock=monotonic();cpu=process_time();paid=0
    with np.load(mpc.BANK) as z:raw=mpc.DeepField(z,'local')
    with mpc.centered(src['ready']):
        a=mpc.arena();cad=mpc.arena(detail='cad');clear=CourtClearance(cad.model,articulated=True)
        camera,rig,stride,lo,hi=mpc.resources(a);tap=rp.old.Tap(camera);axes=a._arm_ids[1][2]
        checker=prior.Reach(cad,src['ready'],src['ready'])
        try:
            a.launch(original['shot'],card=case['card']);a.partner.reset(seed=case['card']['seed']);guarded(a,clear)
            history=[]
            for tick,command in enumerate(original['actions'][:root]):
                frame=camera.capture() if tick%2==0 and tick>=root-32 else None
                history.append((tick,frame,a.joints.copy(),a.data.ctrl[axes].copy()))
                a.exchange(command,mpc.FAR_COMMAND);paid+=1
            state=prior.State(a)
            for mode,record in row['variants'].items():
                state.restore(a);a.card=dict(case['card'],reply_seed=row['random_reply_stream'])
                with patch.object(mpc,'Fragments',lambda f,e:prior.Carried(f,e,practice=[prior.course.PRACTICE],held={case['id']})):
                    if mode=='stock':
                        player=prior.season.Player(mpc.BANK,raw,variant=src['variant'],budget=src['budget'],
                            coverage=prior.fresh.shot.Coverage(rp._read(prior.fresh.RISK)))
                    else:
                        public=dict(ready=src['ready'],variant=src['variant'],budget=src['budget'])
                        player=module.install(raw,checker,public,enabled=mode=='warm',frames=tap)
                try:
                    player.tap=tap;carry=rp._read(prior.season.CARRY)
                    prior.course.resume(player,prior.season.CARRY.with_suffix('').with_suffix('.npz'),carry)
                    observer=prior.Observer(a,rig,player,history);eye=observer.restore(player)
                    restore_readiness_from_actual_images(player,rig,a.control_dt,history)
                    before=player.priors.digest();home=np.asarray(src['ready']);sensed=False
                    differences=[];rgb_differences=[]
                    for at,expected in enumerate(record['controls']):
                        tick=root+at;q=a.joints.copy();ref=a.data.ctrl[axes].copy()
                        frames=tap.capture() if tick%2==0 else None;x=mpc.observed(eye,tick,frames,q)
                        old=record['observations'][at]
                        oldx=old['x'];x_equal=(x is None and oldx is None) or (x is not None and oldx is not None and np.array_equal(x,oldx))
                        if not x_equal:rgb_differences.append(tick)
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
                        if not np.array_equal(u,expected):differences.append(dict(tick=tick,max_difference=float(np.max(abs(u-expected)))))
                        _,_,done,capped,_=a.exchange(expected,mpc.FAR_COMMAND);paid+=1
                        if (done or capped) and at+1!=len(record['controls']):raise RuntimeError('Step after recorded completion')
                    verify_clearance(a)
                    if player.priors.digest()!=before:raise RuntimeError('Frozen memory changed')
                    fresh=[] if mode=='stock' else player.ready.predictor.estimates
                    checks=[]
                    for signal in record.get('body_trace',[]):
                        if signal['action']!='far optical evidence':continue
                        matches=[p for p in fresh if p['tick']==signal['tick'] and p['gate']==signal['gate']]
                        exact=bool(matches and matches[-1]['distribution'][5]==signal['lane']
                            and matches[-1]['distribution'][6]==signal['estimate_height']
                            and matches[-1]['distribution'][-1]==signal['risk'])
                        checks.append(dict(tick=signal['tick'],gate=signal['gate'],fresh_match=exact))
                    out=dict(mode=mode,exact_physical_events=prior.digest_json(a.events)==record['event_sha256'],
                        exact_fresh_policy_actions=not differences,action_differences=differences,
                        exact_rgb_estimates=not rgb_differences,rgb_different_ticks=rgb_differences,
                        recorded_far_signal_checks=checks,all_far_signals_from_fresh_variant=all(p['fresh_match'] for p in checks),
                        fresh_risk_predictions=fresh,clearance=a.clearance_result(),actual_replayed_control_ticks=len(record['controls']),
                        legal=record['legal'],answered=record['answered'])
                    doc['rows'].append(out);doc.update(actual_control_ticks=paid,wall_seconds=monotonic()-clock,cpu_seconds=process_time()-cpu)
                    rp._write(output,plain(doc));print(mode,out['exact_physical_events'],out['exact_fresh_policy_actions'],out['all_far_signals_from_fresh_variant'],flush=True)
                finally:player.close()
        finally:camera.close();a.close();cad.close()
    doc['complete']=True;rp._write(output,plain(doc))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--index',type=int,default=1);args=p.parse_args();run(args.source,args.output,args.index)
