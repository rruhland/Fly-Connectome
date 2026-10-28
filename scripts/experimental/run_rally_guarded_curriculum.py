"""Three sequential proof-backed repairs, followed by one fixed-head bank check.

The admitted bank supplies observed successes, not hidden-state labels. Every
new target rolls in under the latest readout; accepted complete reward witnesses
join the action guards. Intermediate whole-bank outcomes are not invented.
"""
import argparse,json,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from fly_connectome.control.policy import FrozenLinearPolicy
from rally_action_equivalence import certify_episode
from rally_curriculum_acquire import acquire
from pong_raw_guarded_repair import fit
from rally_witness_replay import select_saved,save_replay
from run_rally_incremental_cycle import dataset
from run_rally_raw_optimizer import native_task,more
from return_metrics import aggregate
import run_practical_pong_v2 as base

SLOTS=[('opening','right'),('later','left'),('later','right')]


def earned_cases(receipt,feature,minimum_hits=0):
    X=sp.load_npz(feature);actions=np.array([r['action'] for r in receipt['records']],dtype=np.int64);assert X.shape==(len(actions),4389);start=0;out=[];phase='left' if base.leftward(receipt['seed']) else 'right'
    for ordinal,end in enumerate(i for i,r in enumerate(receipt['records']) if r['hits']):
        if ordinal>=minimum_hits:out.append(dict(seed=receipt['seed'],group=('first_' if ordinal==0 else 'later_')+phase,X=X[start:end+1],actions=actions[start:end+1]))
        start=end+1
    return out


def exact_actions(W,X,actions):
    policy=FrozenLinearPolicy(W);changed=[]
    for i,a in enumerate(actions):
        lo,hi=X.indptr[i:i+2];got,_=policy.choose(X.indices[lo:hi],X.data[lo:hi])
        if got!=a:changed.append(i)
    return changed


def earned_target_witness(directory,receipt,known_hits,role,phase):
    """Keep the ACTUAL learned successful prefix; never its failed suffix."""
    if receipt['report']['hits']<=known_hits:return None
    ends=[i for i,r in enumerate(receipt['records']) if r['hits']];assert len(ends)==receipt['report']['hits']
    end=ends[-1]+1;X=sp.load_npz(Path(directory)/receipt['feature_file']);assert X.shape[0]==len(receipt['records'])
    (Path(directory)/'witnesses').mkdir();sp.save_npz(Path(directory)/'witnesses/actual.npz',X[:end]);np.save(Path(directory)/'witnesses/actions.npy',np.array([r['action'] for r in receipt['records'][:end]],dtype=np.int64))
    return dict(seed=receipt['seed'],role=role,phase=phase,path='witnesses/actual.npz',actions_path='witnesses/actions.npy',hit_indices=ends,target_hits=receipt['report']['hits'],source='actual native learned-policy return, no forced actions',native_receipt=str(Path(directory)/'native_candidate/episodes/train_00.json'))


def check_target(step,seed,known_hits,role,phase,bank,entry,baseline):
    directory=step/'target-native';directory.mkdir();(directory/'proposal.npy').write_bytes((step/'proposal.npy').read_bytes());(directory/'radial-cap.json').write_bytes(Path(bank['radial_cap']).read_bytes())
    base.atomic_json(directory/'manifest.json',dict(source=base.source_signature(),weights_sha256=base.sha(directory/'proposal.npy'),config=baseline['config'],known_current_hits=known_hits,no_forced_actions=True))
    sourceX=sp.load_npz(entry['raw_feature']);certificate=certify_episode(baseline,sourceX,np.load(bank['current_head']),np.load(directory/'proposal.npy'),source_weights_sha256=bank['current_head_sha256'],source_fingerprint=bank['source'],candidate_fingerprint=base.source_signature(),candidate_config=baseline['config'])
    base.atomic_json(directory/'source-episode-certificate.json',{k:v for k,v in certificate.items() if k!='records'})
    if certificate['certified'] and baseline['report']['hits']<=known_hits:
        result=dict(accepted=False,reason='exact completed episode proves no additional known return',transition_cameras=0,initial_cameras=0,wall_seconds=0.,known_current_hits=known_hits);base.atomic_json(directory/'summary.json',result);return result,None
    for name in ('episodes','features'):(directory/'native_candidate'/name).mkdir(parents=True)
    began=time.monotonic();base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=began+30,seconds=30));base.initialize();receipt=native_task(str(directory),seed,0,began+30)
    witness=earned_target_witness(directory,receipt,known_hits,role,phase)
    if witness is not None:
        assert not exact_actions(np.load(directory/'proposal.npy'),sp.load_npz(directory/witness['path']),np.load(directory/witness['actions_path']))
    result=dict(accepted=witness is not None,reason='actual additional native own return' if witness else 'no additional native own return',known_current_hits=known_hits,actual_hits=receipt['report']['hits'],native_episode_complete=receipt['complete'],transition_cameras=receipt['transitions'],initial_cameras=receipt['initial_cameras'],wall_seconds=time.monotonic()-began,witness=witness,new_guard_excludes_failed_suffix=True)
    base.atomic_json(directory/'summary.json',result);return result,witness


def evaluate(directory,bank,entries,baseline,head,cached_native_root=None,native_seconds=500):
    directory=Path(directory);candidate=np.load(head);sourceW=np.load(bank['current_head']);candidate_hash=base.sha(head);outcomes=[];jobs=[];certificates=[]
    for name in ('episodes','features'):(directory/'native_candidate'/name).mkdir(parents=True)
    cached={}
    if cached_native_root is not None:
        cached_native_root=Path(cached_native_root);assert base.read(cached_native_root/'manifest.json')['source']==base.source_signature()
        cached={base.read(p)['seed']:(p,base.read(p)) for p in (cached_native_root/'native_candidate/episodes').glob('*.json')}
    for i,(entry,r) in enumerate(zip(entries,baseline)):
        if r['seed'] in cached:
            path,value=cached[r['seed']];assert value['complete'] and value['weights_sha256']==candidate_hash and value['config']==r['config']
            feature=Path(value['feature_file']);feature=feature if feature.is_absolute() else cached_native_root/feature
            value=dict(value,feature_file=str(feature),execution='reused_completed_native_same_head',executed_transition_cameras=0,executed_initial_cameras=0,source_receipt=str(path),source_receipt_sha256=base.sha(path),source_feature_file=str(feature))
            base.atomic_json(directory/'native_candidate/episodes'/f'train_{i:02d}.json',value);outcomes.append(value);certificates.append(dict(seed=r['seed'],certified=True,reason='previous completed actual native game under identical candidate/config/source',executed_transition_cameras=0));continue
        X=sp.load_npz(entry['raw_feature']);cert=certify_episode(r,X,sourceW,candidate,source_weights_sha256=bank['current_head_sha256'],source_fingerprint=bank['source'],candidate_fingerprint=base.source_signature(),candidate_config=r['config'])
        certificates.append(dict(seed=r['seed'],**{k:v for k,v in cert.items() if k!='records'}))
        if not cert['certified']:jobs.append((i,r['seed']));continue
        value=dict(r,weights_sha256=candidate_hash,records=cert['records'],feature_file=entry['raw_feature'],feature_format='actual capped raw CSR',execution='reused_by_exact_action_certificate',executed_transition_cameras=0,executed_initial_cameras=0,source_receipt=entry['receipt'],source_feature_file=entry['raw_feature'],source_weights_sha256=bank['current_head_sha256'],action_equivalence_certificate={k:v for k,v in cert.items() if k!='records'})
        base.atomic_json(directory/'native_candidate/episodes'/f'train_{i:02d}.json',value);outcomes.append(value)
    base.atomic_json(directory/'reuse-certificates.json',dict(certificates=certificates,reused_complete_games=len(outcomes),reused_logical_cameras=sum(r['transitions'] for r in outcomes),new_observations=0))
    began=time.monotonic();deadline=began+native_seconds;base.atomic_json(directory/'native-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=native_seconds));new=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        pending=iter(jobs);active={}
        def fill():
            while len(active)<4 and time.monotonic()<deadline-2:
                try:i,seed=next(pending)
                except StopIteration:break
                active[pool.submit(native_task,str(directory),seed,i,deadline-1)]=(i,seed)
        fill()
        while active:
            done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for f in done:active.pop(f);r=f.result();new.append(r);outcomes.append(r)
            base.atomic_json(directory/'progress.json',dict(phase='final_bank',logical_games=len(outcomes),new_games=len(new),new_cameras=sum(r['transitions'] for r in new)));fill()
    return outcomes,dict(newly_executed_games=len(new),new_transition_cameras=sum(r['transitions'] for r in new),new_initials=sum(r['initial_cameras'] for r in new),reused_games=len(outcomes)-len(new),reused_completed_native_games=sum(r.get('execution')=='reused_completed_native_same_head' for r in outcomes),reused_logical_cameras=sum(r['transitions'] for r in outcomes if r.get('execution','').startswith('reused_')),wall_seconds=time.monotonic()-began)


def run(directory,bank_path,attempts=3,target_accepted=3,slot_cycle=None,skip_seeds=(),early_stop=False,resume=False,guard_margin_scale=1.,saved_acquisitions=(),cross_witness_warm=False,native_target_fallback=False,feature_metric_path=None,spatial_smoothness=False):
    directory=Path(directory);bank_path=Path(bank_path);bank=base.read(bank_path);entries=bank['bank'];assert entries and len({e['seed'] for e in entries})==len(entries)
    assert bank['source']==base.source_signature() and base.sha(Path(bank['current_head']))==bank['current_head_sha256']
    baseline=[base.read(Path(e['receipt'])) for e in entries];byseed={r['seed']:r for r in baseline};assert all(r['complete'] and r['weights_sha256']==bank['current_head_sha256'] for r in baseline)
    directory.mkdir(exist_ok=True)
    if not resume:
        assert not (directory/'manifest.json').exists();(directory/'carry.npy').write_bytes(Path(bank['current_head']).read_bytes());(directory/'radial-cap.json').write_bytes(Path(bank['radial_cap']).read_bytes())
    else:assert (directory/'manifest.json').exists() and not (directory/'summary.json').exists()
    current_path=directory/'carry.npy';W=np.load(current_path);guards=[]
    feature_scale=None
    if feature_metric_path is not None:
        metric_path=Path(feature_metric_path)
        if not (directory/'feature-scale.npy').exists():(directory/'feature-scale.npy').write_bytes(metric_path.read_bytes())
        assert base.sha(directory/'feature-scale.npy')==base.sha(metric_path);feature_scale=np.load(directory/'feature-scale.npy');assert feature_scale.shape==(W.shape[1],)
    for entry,r in zip(entries,baseline):guards+=earned_cases(r,entry['raw_feature'])
    assert len(guards)==sum(r['report']['hits'] for r in baseline);files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_guarded_curriculum.py','rally_curriculum_acquire.py','pong_raw_guarded_repair.py','run_rally_raw_optimizer.py','rally_action_equivalence.py')}
    manifest=dict(source=base.source_signature(),files=files,bank_path=str(bank_path),bank_sha256=base.sha(bank_path),carry_sha256=base.sha(current_path),slots=slot_cycle or SLOTS,maximum_attempts=attempts,target_accepted=target_accepted,skipped_seeds=list(skip_seeds),early_feasible_stop=early_stop,acquisition_seconds_per_slot=40,solver_seconds_per_slot=90,final_native_seconds=500,planned_phase_seconds_upper=attempts*130+500,no_holdout_fitted=True,no_backprop=True,acceptance='exact all old guard choices and entire native-reward witness action sequence; no intermediate full-bank result assumed')
    manifest['cross_witness_warm_start']=cross_witness_warm
    if spatial_smoothness:
        files['pong_raw_spatial_prior.py']=base.sha(Path(__file__).with_name('pong_raw_spatial_prior.py'));manifest['spatial_smoothness']='fixed nearest-neighbor within existing visual grid channels; no architecture/input change'
    if feature_metric_path is not None:manifest['feature_metric']=dict(source=str(feature_metric_path),sha256=base.sha(directory/'feature-scale.npy'),changes_optimizer_geometry_only=True)
    if native_target_fallback:
        manifest['native_target_fallback']=dict(seconds=30,acceptance='all current earned choices retained plus actual additional own return; guard the actual successful prefix only')
        manifest['acceptance']='exact all old earned choices AND either full rewarded-witness action certificate OR additional native target return'
        manifest['planned_phase_seconds_upper']=attempts*160+500
    if saved_acquisitions or guard_margin_scale!=1.:
        files['rally_witness_replay.py']=base.sha(Path(__file__).with_name('rally_witness_replay.py'))
        manifest.update(guard_margin_scale=guard_margin_scale,saved_acquisitions=list(map(str,saved_acquisitions)),exact_conflict_screen=True,cached_witnesses_prioritized_by='fewest current-contrasting actions, remaining forced commands, seed; actual whole trajectory certificate required')
    if resume:
        original=base.read(directory/'manifest.json')
        assert json.dumps({k:v for k,v in original.items() if k!='files'},sort_keys=True)==json.dumps({k:v for k,v in manifest.items() if k!='files'},sort_keys=True)
        assert all(original['files'][k]==v for k,v in files.items() if k!=Path(__file__).name)
        restarts=base.read(directory/'restart-record.json')['restarts'] if (directory/'restart-record.json').exists() else []
        restarts.append(dict(resumed_wall_ns=time.time_ns(),original_manifest_sha256=base.sha(directory/'manifest.json'),files=files,reason='environment reset interrupted numerical fit; completed acquisitions and steps reused; original guard margins unchanged'))
        base.atomic_json(directory/'restart-record.json',dict(restarts=restarts));manifest=original
    else:base.atomic_json(directory/'manifest.json',manifest)
    if not resume:base.atomic_json(directory/'curriculum-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=time.monotonic()))
    original_clock=base.read(directory/'curriculum-clock.json');began=time.monotonic()-(time.time_ns()-original_clock['started_wall_ns'])/1e9
    steps=[];accepted=[];excluded=list(skip_seeds);warm_rows=[];warm_duals={};normal0=base.read(base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json');normal={k:normal0[k] for k in ('input_rms','kappa')}
    plan=slot_cycle or SLOTS;used_saved=set();known_hits={seed:r['report']['hits'] for seed,r in byseed.items()};entry_byseed={e['seed']:e for e in entries}
    for number in range(1,attempts+1):
        if len(accepted)>=target_accepted:break
        role,phase=plan[(number-1)%len(plan)]
        step=directory/f'step_{number}';step.mkdir(exist_ok=True);acqdir=step/'acquisition'
        if (step/'before.npy').exists():assert base.sha(step/'before.npy')==base.sha(current_path)
        else:(step/'before.npy').write_bytes(current_path.read_bytes())
        if (acqdir/'summary.json').exists():
            am=base.read(acqdir/'manifest.json');assert am['current_head_sha256']==base.sha(current_path) and am['bank_sha256']==base.sha(bank_path);acquisition=base.read(acqdir/'summary.json')
        else:
            saved,screen=select_saved(saved_acquisitions,W,guards,excluded,used_saved,base.source_signature(),known_hits,{role for role,phase in plan}) if saved_acquisitions else (None,[])
            base.atomic_json(step/'cached-witness-screen.json',dict(screen=screen,new_observations=0))
            if saved is not None:acquisition=save_replay(acqdir,*saved,current_path,bank_path)
            else:acquisition=acquire(acqdir,bank_path,current_path,role,phase,40,excluded)
        assert acquisition['source_frozen']
        for item in acquisition['results']:
            if item.get('reason') and item['seed'] not in excluded:excluded.append(item['seed'])
        # Newly observed current roll-ins may have advanced beyond the admitted
        # head. Preserve those actual extra earned segments, even if no fit follows.
        for r in acquisition['results']:
            if r['rollin_origin']=='new current-head native roll-in':
                receipt=base.read(Path(r['rollin_receipt']));guards+=earned_cases(receipt,r['rollin_raw_feature'],minimum_hits=byseed[r['seed']]['report']['hits']);known_hits[r['seed']]=max(known_hits[r['seed']],receipt['report']['hits'])
        if saved_acquisitions and not acquisition.get('off_policy_own_reward_replay'):
            chosen,screen=select_saved([acqdir],W,guards,excluded,used_saved,base.source_signature(),known_hits,{role for role,phase in plan});base.atomic_json(step/'new-witness-screen.json',dict(screen=screen,new_observations=0));acquisition=dict(acquisition,selected_witness=chosen[1] if chosen else None)
        witness=acquisition['selected_witness']
        if witness is not None:role,phase=witness['role'],witness['phase']
        record=dict(step=number,role=role,phase=phase,acquisition_cameras=acquisition['transition_cameras'],acquisition_initials=acquisition['initial_cameras'],acquisition_seconds=acquisition['wall_seconds'],accepted=False,reward_witness_reused=acquisition.get('off_policy_own_reward_replay',False))
        if (step/'summary.json').exists():
            assert resume;record=base.read(step/'summary.json');steps.append(record)
            if witness is not None:
                stable_rows=sum(c['X'].shape[0] for c in guards)
                with np.load(step/'solver-state.npz') as state:
                    use=state['working_rows']<2*stable_rows;warm_rows=state['working_rows'][use].astype(int).tolist();warm_duals={int(i):float(y) for i,y in zip(state['working_rows'][use],state['guard_duals'][use])}
                excluded.append(witness['seed'])
                if record['accepted']:
                    actual=record.get('target_native',{}).get('witness');accepted_root=step/'target-native' if actual else acqdir;accepted_witness=actual or witness
                    X=sp.load_npz(accepted_root/accepted_witness['path']);actions=np.load(accepted_root/accepted_witness['actions_path']);current_path=step/'proposal.npy';W=np.load(current_path)
                    assert not exact_actions(W,X,actions)
                    guards.append(dict(seed=witness['seed'],group=role+'_'+phase,X=X,actions=actions));accepted.append(dict(step=number,seed=witness['seed'],target_hits=accepted_witness['target_hits'],witness_root=str(accepted_root),witness=accepted_witness,head=str(current_path)));known_hits[witness['seed']]=max(known_hits[witness['seed']],accepted_witness['target_hits'])
            print(json.dumps(dict(reused_completed_step=number,accepted=record['accepted'])),flush=True);continue
        if witness is None:record['reason']='no rewarded current-head recovery within acquisition budget';steps.append(record);base.atomic_json(step/'summary.json',record);continue
        positive,unused,matching=dataset(acqdir,W,[],[witness]);assert sum(c['X'].shape[0] for c in positive)>0
        stable_rows=sum(c['X'].shape[0] for c in guards);fit_began=time.monotonic();fit_restart={}
        if (step/'fit-clock.json').exists():
            assert resume;old=base.read(step/'fit-clock.json');archive=step/f'fit-clock.interrupted-{len(restarts)}.json';assert not archive.exists();archive.write_bytes((step/'fit-clock.json').read_bytes())
            fit_restart=dict(interrupted_clock=str(archive),elapsed_wall_since_interrupted_start_seconds=(time.time_ns()-old['started_wall_ns'])/1e9,includes_environment_outage=True,repeated_acquisition_cameras=0)
        base.atomic_json(step/'fit-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=fit_began,deadline_monotonic=fit_began+90,**fit_restart))
        delta,report=fit(W,positive,guards+matching,normal,fit_began+88,ridge=1e-6,batch=512,initial_rows=warm_rows if cross_witness_warm else (),initial_guard_duals=warm_duals if cross_witness_warm else {},state_path=step/'solver-state.npz',stop_on_earned_repair=early_stop,guard_margin_scale=guard_margin_scale,feature_scale=feature_scale,spatial_smoothness=spatial_smoothness)
        base.save_array(step/'delta.npy',delta);base.save_array(step/'proposal.npy',W+delta);base.atomic_json(step/'fit-report.json',report)
        X=sp.load_npz(acqdir/witness['path']);actions=np.load(acqdir/witness['actions_path']);changed=exact_actions(W+delta,X,actions);certificate=dict(certified=not changed,changed_rows=changed,checked_rows=len(actions),seed=witness['seed'],native_reward_source=str(acqdir/'summary.json'),new_observations=0);base.atomic_json(step/'whole-witness-certificate.json',certificate)
        with np.load(step/'solver-state.npz') as state:
            use=state['working_rows']<2*stable_rows;warm_rows=state['working_rows'][use].astype(int).tolist();warm_duals={int(i):float(y) for i,y in zip(state['working_rows'][use],state['guard_duals'][use])}
        record.update(witness_seed=witness['seed'],positive_rows=report['positive_rows'],fit_seconds=report['wall_seconds'],fit_status=report['status'],requested_actions_fraction=report['positive']['requested_action_fraction'],full_witness_certified=certificate['certified'])
        accepted_witness=witness;accepted_root=acqdir;accepted_by='whole rewarded witness action certificate' if certificate['certified'] else None
        if not certificate['certified'] and native_target_fallback:
            assert report['guard_action_changes']==0
            target,actual_witness=check_target(step,witness['seed'],known_hits[witness['seed']],role,phase,bank,entry_byseed[witness['seed']],byseed[witness['seed']]);record.update(target_native=target)
            if actual_witness is not None:
                accepted_witness=actual_witness;accepted_root=step/'target-native';X=sp.load_npz(accepted_root/actual_witness['path']);actions=np.load(accepted_root/actual_witness['actions_path']);accepted_by='actual additional native return with exact old earned choices'
        if accepted_by is not None:
            record['accepted']=True;record['accepted_by']=accepted_by;current_path=step/'proposal.npy';W=np.load(current_path);guards.append(dict(seed=accepted_witness['seed'],group=role+'_'+phase,X=X,actions=actions));accepted.append(dict(step=number,seed=accepted_witness['seed'],target_hits=accepted_witness['target_hits'],witness_root=str(accepted_root),witness=accepted_witness,head=str(current_path)));known_hits[witness['seed']]=max(known_hits[witness['seed']],accepted_witness['target_hits']);excluded.append(witness['seed'])
        else:
            record['reason']='bounded QP did not reproduce whole recovery; carried head unchanged'
            excluded.append(witness['seed'])
        steps.append(record);base.atomic_json(step/'summary.json',record);base.atomic_json(directory/'learning-steps.json',dict(steps=steps,accepted=accepted,current_head=str(current_path),whole_bank_not_yet_evaluated=True));print(json.dumps(dict(step_result=record)),flush=True)
    if not accepted:
        base.atomic_json(directory/'summary.json',dict(status='no_accepted_updates',steps=steps,carry_unchanged=True,accepted_steps=0,whole_bank_skipped=True,wall_seconds=time.monotonic()-began,acquisition_transition_cameras=sum(s['acquisition_cameras'] for s in steps),acquisition_initials=sum(s['acquisition_initials'] for s in steps),target_transition_cameras=sum(s.get('target_native',{}).get('transition_cameras',0) for s in steps),target_initials=sum(s.get('target_native',{}).get('initial_cameras',0) for s in steps)));return
    (directory/'proposal.npy').write_bytes(current_path.read_bytes());outcomes,native=evaluate(directory,bank,entries,baseline,directory/'proposal.npy');result_byseed={r['seed']:r for r in outcomes};violations=[r['seed'] for r in outcomes if r['complete'] and r['report']['hits']<byseed[r['seed']]['report']['hits']]
    first_acq=base.read(directory/'step_1/acquisition/work-clock.json')['started_monotonic']
    summary=dict(status='complete' if len(outcomes)==len(entries) and all(r['complete'] for r in outcomes) else 'bounded_partial',steps=steps,accepted_steps=len(accepted),baseline=aggregate([r['annotation'] for r in baseline]),candidate=aggregate([r['annotation'] for r in outcomes]),baseline_more=more(baseline),candidate_more=more(outcomes),guarded_native_return_violations=violations,restart_record=str(directory/'restart-record.json') if resume else None,
        targeted_native_outcomes=[dict(seed=w['seed'],at_least_earned_hits=w['target_hits'],baseline_hits=byseed[w['seed']]['report']['hits'],candidate_hits=result_byseed[w['seed']]['report']['hits'] if w['seed'] in result_byseed else None) for w in accepted],
        acquisition_transition_cameras=sum(s['acquisition_cameras'] for s in steps),acquisition_initials=sum(s['acquisition_initials'] for s in steps),target_transition_cameras=sum(s.get('target_native',{}).get('transition_cameras',0) for s in steps),target_initials=sum(s.get('target_native',{}).get('initial_cameras',0) for s in steps),native=native,wall_seconds_since_curriculum_start=time.monotonic()-began,wall_seconds_since_first_acquisition=time.monotonic()-first_acq,
        timed_phase_seconds=sum(s['acquisition_seconds']+s.get('fit_seconds',0)+s.get('target_native',{}).get('wall_seconds',0) for s in steps)+native['wall_seconds'],carry_immutable=base.sha(directory/'carry.npy')==manifest['carry_sha256'],source_frozen=manifest['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files},proposal_sha256=base.sha(directory/'proposal.npy'),historical_evaluation_now_training=True,no_fresh_evaluation=True,no_automatic_promotion=True)
    base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--bank',required=True);p.add_argument('--attempts',type=int,default=3);p.add_argument('--target-accepted',type=int,default=3);p.add_argument('--larger-rotation',action='store_true');p.add_argument('--later-only',action='store_true');p.add_argument('--skip-seed',type=int,action='append',default=[]);p.add_argument('--early-stop',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--guard-margin-scale',type=float,default=1.);p.add_argument('--saved-acquisition',action='append',default=[]);p.add_argument('--cross-witness-warm',action='store_true',help='legacy opt-in; default starts each new witness with an empty constraint set and zero duals');p.add_argument('--native-target-fallback',action='store_true');p.add_argument('--feature-metric');p.add_argument('--smooth-spatial-residual',action='store_true');a=p.parse_args();assert 1<=a.target_accepted<=a.attempts<=8;plan=[('later','left'),('later','right')] if a.later_only else [('opening','left'),('later','right'),('opening','right'),('later','left')] if a.larger_rotation else None;run(a.directory,a.bank,a.attempts,a.target_accepted,plan,a.skip_seed,a.early_stop,a.resume,a.guard_margin_scale,a.saved_acquisition,a.cross_witness_warm,a.native_target_fallback,a.feature_metric,a.smooth_spatial_residual)
