"""One minimal raw repair protected by all current earned action prefixes.

Unchanged complete episodes are reused only with exact action certificates;
logical camera lengths and newly executed policy observations are separate.
"""
import argparse,json,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from pong_raw_guarded_repair import fit
from run_rally_incremental_cycle import current_rows,dataset
from run_rally_raw_optimizer import native_task,more
from rally_action_equivalence import certify_episode
from return_metrics import aggregate
from fly_connectome.control.policy import FrozenLinearPolicy
import run_practical_pong_v2 as base


def run(directory,source,witness_root,seed,resume=False,initial_delta_file=None):
    directory=Path(directory);source=Path(source);witness_root=Path(witness_root);assert resume or not directory.exists();W,rows=current_rows(source)
    witness=next(w for w in base.read(witness_root/'discovery-summary.json')['selected_witnesses'] if w['seed']==seed)
    assert base.read(witness_root/'manifest.json')['carry_sha256']==base.sha(source/'proposal.npy')
    positive,guards,matching=dataset(witness_root,W,rows,[witness]);assert sum(c['X'].shape[0] for c in positive)==3
    directory.mkdir(exist_ok=resume)
    for old,new in [('proposal.npy','carry.npy'),('radial-cap.json','radial-cap.json')]:(directory/new).write_bytes((source/old).read_bytes())
    files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_guarded_repair.py','pong_raw_guarded_repair.py','run_rally_incremental_cycle.py','run_rally_raw_optimizer.py','rally_action_equivalence.py')}
    manifest=dict(source=base.source_signature(),files=files,source_root=str(source),carry_sha256=base.sha(directory/'carry.npy'),witness_root=str(witness_root),witness=witness,guarded_earned_segments=len(guards),matching_witness_guard_rows=sum(c['X'].shape[0] for c in matching),rows=[r['receipt'] for r in rows],work_seconds=420,solver_seconds=90,
        objective='diagonal raw contrast ridge plus requested action hinge slack; every earned prefix has capped action-choice margins',required_guard_gap='min(kappa, half old gap)',failed_suffix_guarded=False,no_new_capture=True,no_new_discovery=True,no_holdout_fitted=True,no_backprop=True)
    if resume:
        original=base.read(directory/'manifest.json');clock=base.read(directory/'work-clock.json');assert original['carry_sha256']==manifest['carry_sha256'] and original['source']==manifest['source']
        began=clock['started_monotonic'];deadline=clock['deadline_monotonic'];base.atomic_json(directory/('solver-warm-continuation.json' if initial_delta_file else 'matching-guard-amendment.json'),dict(reason='add samewitness matching actions to exact fulltrajectory certificate; fix numpy integer report serialization; no newcapture or clockreset',original_manifest=original,resumed_manifest=manifest,original_clock=clock,resumed_wall_ns=time.time_ns()))
    else:
        base.atomic_json(directory/'manifest.json',manifest);began=time.monotonic();deadline=began+420;base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline))
    normal0=base.read(base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json');normal={k:normal0[k] for k in ('input_rms','kappa')}
    initial_delta=np.load(initial_delta_file) if initial_delta_file else None
    delta,report=fit(W,positive,guards+matching,normal,min(deadline-90,time.monotonic()+60 if initial_delta_file else time.monotonic()+90),ridge=1e-6,initial_delta=initial_delta,state_path=directory/'solver-state.npz');base.save_array(directory/'applied_delta.npy',delta);base.save_array(directory/'proposal.npy',W+delta);base.atomic_json(directory/'fit-report.json',report);print(json.dumps(dict(fit=report)),flush=True)
    WX=sp.load_npz(witness_root/witness['path']);wa=np.load(witness_root/witness['actions_path']);policy=FrozenLinearPolicy(W+delta);witness_changes=[]
    for i,a in enumerate(wa):
        lo,hi=WX.indptr[i:i+2];action,_=policy.choose(WX.indices[lo:hi],WX.data[lo:hi])
        if action!=a:witness_changes.append(i)
    base.atomic_json(directory/'full-witness-action-certificate.json',dict(certified=not witness_changes,seed=seed,checked_rows=len(wa),changed_rows=witness_changes,native_reward_source=str(witness_root/'discovery-summary.json'),executed_transition_cameras=0,reason='exact greedy actions along whole reward-earned trajectory imply reproduced trajectory with same deterministic sensor/physics path'))
    if witness_changes:
        base.atomic_json(directory/'summary.json',dict(status='no_full_witness_repair',zero_candidate_observations=True,fit=report,wall_seconds=time.monotonic()-began));return
    for name in ('episodes','features'):(directory/'native_candidate'/name).mkdir(parents=True)
    candidate=W+delta;candidate_hash=base.sha(directory/'proposal.npy');source_manifest=base.read(source/'manifest.json');outcomes=[];jobs=[];certificates=[];cert_began=time.monotonic()
    for i,r in enumerate(rows):
        X=sp.load_npz(r['raw_feature']);cert=certify_episode(r,X,W,candidate,source_weights_sha256=manifest['carry_sha256'],source_fingerprint=source_manifest['source'],candidate_fingerprint=manifest['source'],candidate_config=r['config'])
        certificates.append(dict(seed=r['seed'],**{k:v for k,v in cert.items() if k!='records'}))
        if not cert['certified']:jobs.append((i,r['seed']));continue
        # This is a provenance-bearing reuse receipt, never a claimed new rollout.
        value=dict(r,weights_sha256=candidate_hash,records=cert['records'],feature_file=r['raw_feature'],feature_format='actual capped raw CSR',executed_transition_cameras=0,executed_initial_cameras=0,
            execution='reused_by_exact_action_certificate',source_receipt=r['receipt'],source_feature_file=r['raw_feature'],source_weights_sha256=manifest['carry_sha256'],action_equivalence_certificate={k:v for k,v in cert.items() if k!='records'})
        value.pop('receipt',None);value.pop('raw_feature',None)
        base.atomic_json(directory/'native_candidate/episodes'/f'train_{i:02d}.json',value);outcomes.append(value)
    base.atomic_json(directory/'reuse-certificates.json',dict(certificates=certificates,certified_games=len(outcomes),reused_logical_transition_cameras=sum(r['transitions'] for r in outcomes),executed_transition_cameras=0,wall_seconds=time.monotonic()-cert_began))
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
            for f in done:active.pop(f);outcomes.append(f.result())
            base.atomic_json(directory/'progress.json',dict(phase='native96',logical_games=len(outcomes),newly_executed_games=len(outcomes)-(len(rows)-len(jobs)),newly_executed_cameras=sum(r['transitions'] for r in outcomes if r.get('execution')!='reused_by_exact_action_certificate')));fill()
    byseed={r['seed']:r for r in outcomes};baseline={r['seed']:r for r in rows};executed=[r for r in outcomes if r.get('execution')!='reused_by_exact_action_certificate'];reused=[r for r in outcomes if r.get('execution')=='reused_by_exact_action_certificate']
    retention_violations=[r['seed'] for r in outcomes if r['report']['hits']<baseline[r['seed']]['report']['hits']]
    summary=dict(status='complete' if len(outcomes)==96 and all(r['complete'] for r in outcomes) else 'bounded_partial',baseline=aggregate([r['annotation'] for r in rows]),candidate=aggregate([r['annotation'] for r in outcomes]),baseline_more=more(rows),candidate_more=more(outcomes),
        repair_outcomes=[dict(seed=seed,role=witness['role'],baseline_hits=baseline[seed]['report']['hits'],candidate_hits=byseed[seed]['report']['hits'] if seed in byseed else None)],guarded_native_return_violations=retention_violations,
        newly_executed_games=len(executed),reused_complete_games=len(reused),native_transition_cameras=sum(r['transitions'] for r in executed),native_initials=sum(r['initial_cameras'] for r in executed),logical_transition_cameras=sum(r['transitions'] for r in outcomes),reused_logical_transition_cameras=sum(r['transitions'] for r in reused),wall_seconds=time.monotonic()-began,
        carry_immutable=base.sha(directory/'carry.npy')==manifest['carry_sha256'],source_frozen=files=={n:base.sha(Path(__file__).with_name(n)) for n in files} and manifest['source']==base.source_signature(),proposal_sha256=candidate_hash,no_fresh_evaluation=True,no_automatic_promotion=True)
    base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--source-root',required=True);p.add_argument('--witness-root',required=True);p.add_argument('--seed',type=int,default=104562292);p.add_argument('--resume',action='store_true');p.add_argument('--initial-delta');a=p.parse_args();run(a.directory,a.source_root,a.witness_root,a.seed,a.resume,a.initial_delta)
