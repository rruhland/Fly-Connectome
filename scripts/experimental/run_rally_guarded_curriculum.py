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


def evaluate(directory,bank,entries,baseline,head):
    directory=Path(directory);candidate=np.load(head);sourceW=np.load(bank['current_head']);candidate_hash=base.sha(head);outcomes=[];jobs=[];certificates=[]
    for name in ('episodes','features'):(directory/'native_candidate'/name).mkdir(parents=True)
    for i,(entry,r) in enumerate(zip(entries,baseline)):
        X=sp.load_npz(entry['raw_feature']);cert=certify_episode(r,X,sourceW,candidate,source_weights_sha256=bank['current_head_sha256'],source_fingerprint=bank['source'],candidate_fingerprint=base.source_signature(),candidate_config=r['config'])
        certificates.append(dict(seed=r['seed'],**{k:v for k,v in cert.items() if k!='records'}))
        if not cert['certified']:jobs.append((i,r['seed']));continue
        value=dict(r,weights_sha256=candidate_hash,records=cert['records'],feature_file=entry['raw_feature'],feature_format='actual capped raw CSR',execution='reused_by_exact_action_certificate',executed_transition_cameras=0,executed_initial_cameras=0,source_receipt=entry['receipt'],source_feature_file=entry['raw_feature'],source_weights_sha256=bank['current_head_sha256'],action_equivalence_certificate={k:v for k,v in cert.items() if k!='records'})
        base.atomic_json(directory/'native_candidate/episodes'/f'train_{i:02d}.json',value);outcomes.append(value)
    base.atomic_json(directory/'reuse-certificates.json',dict(certificates=certificates,reused_complete_games=len(outcomes),reused_logical_cameras=sum(r['transitions'] for r in outcomes),new_observations=0))
    began=time.monotonic();deadline=began+500;base.atomic_json(directory/'native-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=500));new=[]
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
    return outcomes,dict(newly_executed_games=len(new),new_transition_cameras=sum(r['transitions'] for r in new),new_initials=sum(r['initial_cameras'] for r in new),reused_games=len(outcomes)-len(new),reused_logical_cameras=sum(r['transitions'] for r in outcomes if r.get('execution')=='reused_by_exact_action_certificate'),wall_seconds=time.monotonic()-began)


def run(directory,bank_path,attempts=3,target_accepted=3,slot_cycle=None,skip_seeds=(),early_stop=False):
    directory=Path(directory);bank_path=Path(bank_path);bank=base.read(bank_path);entries=bank['bank'];assert entries and len({e['seed'] for e in entries})==len(entries)
    assert bank['source']==base.source_signature() and base.sha(Path(bank['current_head']))==bank['current_head_sha256']
    baseline=[base.read(Path(e['receipt'])) for e in entries];byseed={r['seed']:r for r in baseline};assert all(r['complete'] and r['weights_sha256']==bank['current_head_sha256'] for r in baseline)
    directory.mkdir(exist_ok=True);assert not (directory/'manifest.json').exists();(directory/'carry.npy').write_bytes(Path(bank['current_head']).read_bytes());(directory/'radial-cap.json').write_bytes(Path(bank['radial_cap']).read_bytes())
    current_path=directory/'carry.npy';W=np.load(current_path);guards=[]
    for entry,r in zip(entries,baseline):guards+=earned_cases(r,entry['raw_feature'])
    assert len(guards)==sum(r['report']['hits'] for r in baseline);files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_guarded_curriculum.py','rally_curriculum_acquire.py','pong_raw_guarded_repair.py','run_rally_raw_optimizer.py','rally_action_equivalence.py')}
    manifest=dict(source=base.source_signature(),files=files,bank_path=str(bank_path),bank_sha256=base.sha(bank_path),carry_sha256=base.sha(current_path),slots=slot_cycle or SLOTS,maximum_attempts=attempts,target_accepted=target_accepted,skipped_seeds=list(skip_seeds),early_feasible_stop=early_stop,acquisition_seconds_per_slot=40,solver_seconds_per_slot=90,final_native_seconds=500,planned_phase_seconds_upper=attempts*130+500,no_holdout_fitted=True,no_backprop=True,acceptance='exact all old guard choices and entire native-reward witness action sequence; no intermediate full-bank result assumed')
    base.atomic_json(directory/'manifest.json',manifest);began=time.monotonic();base.atomic_json(directory/'curriculum-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began));steps=[];accepted=[];excluded=list(skip_seeds);warm_rows=[];warm_duals={};normal0=base.read(base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json');normal={k:normal0[k] for k in ('input_rms','kappa')}
    plan=slot_cycle or SLOTS
    for number in range(1,attempts+1):
        if len(accepted)>=target_accepted:break
        role,phase=plan[(number-1)%len(plan)]
        step=directory/f'step_{number}';step.mkdir(exist_ok=True);acqdir=step/'acquisition';(step/'before.npy').write_bytes(current_path.read_bytes())
        if (acqdir/'summary.json').exists():
            am=base.read(acqdir/'manifest.json');assert am['current_head_sha256']==base.sha(current_path) and am['bank_sha256']==base.sha(bank_path);acquisition=base.read(acqdir/'summary.json')
        else:acquisition=acquire(acqdir,bank_path,current_path,role,phase,40,excluded)
        assert acquisition['source_frozen']
        for item in acquisition['results']:
            if item.get('reason') and item['seed'] not in excluded:excluded.append(item['seed'])
        # Newly observed current roll-ins may have advanced beyond the admitted
        # head. Preserve those actual extra earned segments, even if no fit follows.
        for r in acquisition['results']:
            if r['rollin_origin']=='new current-head native roll-in':
                receipt=base.read(Path(r['rollin_receipt']));guards+=earned_cases(receipt,r['rollin_raw_feature'],minimum_hits=byseed[r['seed']]['report']['hits'])
        witness=acquisition['selected_witness'];record=dict(step=number,role=role,phase=phase,acquisition_cameras=acquisition['transition_cameras'],acquisition_initials=acquisition['initial_cameras'],acquisition_seconds=acquisition['wall_seconds'],accepted=False)
        if witness is None:record['reason']='no rewarded current-head recovery within acquisition budget';steps.append(record);base.atomic_json(step/'summary.json',record);continue
        positive,unused,matching=dataset(acqdir,W,[],[witness]);assert sum(c['X'].shape[0] for c in positive)>0
        stable_rows=sum(c['X'].shape[0] for c in guards);fit_began=time.monotonic();base.atomic_json(step/'fit-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=fit_began,deadline_monotonic=fit_began+90))
        delta,report=fit(W,positive,guards+matching,normal,fit_began+88,ridge=1e-6,batch=512,initial_rows=warm_rows,initial_guard_duals=warm_duals,state_path=step/'solver-state.npz',stop_on_earned_repair=early_stop)
        base.save_array(step/'delta.npy',delta);base.save_array(step/'proposal.npy',W+delta);base.atomic_json(step/'fit-report.json',report)
        X=sp.load_npz(acqdir/witness['path']);actions=np.load(acqdir/witness['actions_path']);changed=exact_actions(W+delta,X,actions);certificate=dict(certified=not changed,changed_rows=changed,checked_rows=len(actions),seed=witness['seed'],native_reward_source=str(acqdir/'summary.json'),new_observations=0);base.atomic_json(step/'whole-witness-certificate.json',certificate)
        with np.load(step/'solver-state.npz') as state:
            use=state['working_rows']<2*stable_rows;warm_rows=state['working_rows'][use].astype(int).tolist();warm_duals={int(i):float(y) for i,y in zip(state['working_rows'][use],state['guard_duals'][use])}
        record.update(witness_seed=witness['seed'],positive_rows=report['positive_rows'],fit_seconds=report['wall_seconds'],fit_status=report['status'],requested_actions_fraction=report['positive']['requested_action_fraction'],full_witness_certified=certificate['certified'])
        if certificate['certified']:
            record['accepted']=True;current_path=step/'proposal.npy';W=np.load(current_path);guards.append(dict(seed=witness['seed'],group=role+'_'+phase,X=X,actions=actions));accepted.append(dict(step=number,seed=witness['seed'],target_hits=witness['target_hits'],witness_root=str(acqdir),witness=witness,head=str(current_path)));excluded.append(witness['seed'])
        else:
            record['reason']='bounded QP did not reproduce whole recovery; carried head unchanged'
            excluded.append(witness['seed'])
        steps.append(record);base.atomic_json(step/'summary.json',record);base.atomic_json(directory/'learning-steps.json',dict(steps=steps,accepted=accepted,current_head=str(current_path),whole_bank_not_yet_evaluated=True));print(json.dumps(dict(step_result=record)),flush=True)
    if not accepted:
        base.atomic_json(directory/'summary.json',dict(status='no_accepted_updates',steps=steps,carry_unchanged=True,accepted_steps=0,whole_bank_skipped=True,wall_seconds=time.monotonic()-began));return
    (directory/'proposal.npy').write_bytes(current_path.read_bytes());outcomes,native=evaluate(directory,bank,entries,baseline,directory/'proposal.npy');result_byseed={r['seed']:r for r in outcomes};violations=[r['seed'] for r in outcomes if r['complete'] and r['report']['hits']<byseed[r['seed']]['report']['hits']]
    first_acq=base.read(directory/'step_1/acquisition/work-clock.json')['started_monotonic']
    summary=dict(status='complete' if len(outcomes)==len(entries) and all(r['complete'] for r in outcomes) else 'bounded_partial',steps=steps,accepted_steps=len(accepted),baseline=aggregate([r['annotation'] for r in baseline]),candidate=aggregate([r['annotation'] for r in outcomes]),baseline_more=more(baseline),candidate_more=more(outcomes),guarded_native_return_violations=violations,
        targeted_native_outcomes=[dict(seed=w['seed'],at_least_earned_hits=w['target_hits'],baseline_hits=byseed[w['seed']]['report']['hits'],candidate_hits=result_byseed[w['seed']]['report']['hits'] if w['seed'] in result_byseed else None) for w in accepted],
        acquisition_transition_cameras=sum(s['acquisition_cameras'] for s in steps),acquisition_initials=sum(s['acquisition_initials'] for s in steps),native=native,wall_seconds_since_curriculum_start=time.monotonic()-began,wall_seconds_since_first_acquisition=time.monotonic()-first_acq,
        timed_phase_seconds=sum(s['acquisition_seconds']+s.get('fit_seconds',0) for s in steps)+native['wall_seconds'],carry_immutable=base.sha(directory/'carry.npy')==manifest['carry_sha256'],source_frozen=manifest['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files},proposal_sha256=base.sha(directory/'proposal.npy'),historical_evaluation_now_training=True,no_fresh_evaluation=True,no_automatic_promotion=True)
    base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--bank',required=True);p.add_argument('--attempts',type=int,default=3);p.add_argument('--target-accepted',type=int,default=3);p.add_argument('--larger-rotation',action='store_true');p.add_argument('--skip-seed',type=int,action='append',default=[]);p.add_argument('--early-stop',action='store_true');a=p.parse_args();assert 1<=a.target_accepted<=a.attempts<=8;run(a.directory,a.bank,a.attempts,a.target_accepted,[('opening','left'),('later','right'),('opening','right'),('later','left')] if a.larger_rotation else None,a.skip_seed,a.early_stop)
