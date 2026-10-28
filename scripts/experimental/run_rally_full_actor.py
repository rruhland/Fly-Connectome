"""One bounded on-policy full-rally output-learning block; final endpoint only."""
import argparse,json,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from pong_full_rally_actor import task,calibrate_temperature,paired_gradient,capped_step,probabilities
from pong_constrained_residual import CONTRAST
from pong_fixed_rbf import load_readout
from pong_fixed_pairwise import FrozenPairwiseReadout,save_readout
from pong_joint_output import undo
from run_rally_fixed_bank_check import run as native_check
from run_rally_raw_fresh_screen import run as fresh_run
from run_rally_repertoire_collection import exclusions,select_starts
import run_practical_pong_v2 as base

FILES=('run_rally_full_actor.py','pong_full_rally_actor.py','pong_joint_output.py','pong_fixed_rbf.py','pong_fixed_pairwise.py','run_rally_raw_optimizer.py','run_rally_fixed_bank_check.py','run_rally_guarded_curriculum.py','rally_action_equivalence.py','run_rally_raw_fresh_screen.py','run_rally_repertoire_collection.py','rally_raw_capture.py')


def exact_q(policy,X):
    return np.array([policy.choose(X.indices[X.indptr[i]:X.indptr[i+1]],X.data[X.indptr[i]:X.indptr[i+1]])[1] for i in range(X.shape[0])])


def run(directory,spec_path):
    directory=Path(directory);spec_path=Path(spec_path);assert not directory.exists();directory.mkdir();began=time.monotonic();deadline=began+900;train_deadline=began+600;spec=base.read(spec_path);base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,training_deadline_monotonic=train_deadline,seconds=900,training_max_seconds=600,implementation_seconds_separate_estimate=spec['implementation_seconds_separate_estimate']));(directory/'spec.json').write_bytes(spec_path.read_bytes());files={n:base.sha(Path(__file__).with_name(n)) for n in FILES};source=base.source_signature();bank=base.read(spec['bank']);assert base.sha(spec['bank'])==spec['bank_sha256'] and bank['source']==source
    for p,digest in spec['pinned_assets'].items():assert base.sha(p)==digest
    current=load_readout(bank['current_model_bundle']);initial=current;current_root=Path(bank['current_head']).parent;den=np.load(spec['raw_denominator']);phi=spec['phi_denominator'];G=np.load(spec['metric'],mmap_mode='r');L=la.cholesky(G,lower=True,check_finite=False);byseed={e['seed']:e for e in bank['bank']};summary=dict(status='preparing_actor',new_transition_cameras=0,new_initial_cameras=0,training_transition_cameras=0,training_initials=0,accepted_batches=0)
    manifest=dict(source=source,files=files,spec_sha256=base.sha(spec_path),bank_sha256=base.sha(spec['bank']),initial_policy_sha256=bank['current_policy_sha256'],initial_base_sha256=bank['current_head_sha256'],output_only_analytic_gradient=True,no_backprop=True,no_action_locks=True,no_feature_or_architecture_change=True,endpoint_selection='FINAL completed batch only; no TRAIN checkpoint selection',caps_change_gradient_expectation=True)
    base.atomic_json(directory/'manifest.json',manifest)
    def finish():
        summary.update(wall_seconds=time.monotonic()-began,source_frozen=source==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in FILES},assets_unchanged=all(base.sha(p)==d for p,d in spec['pinned_assets'].items()),bank_unchanged=base.sha(spec['bank'])==spec['bank_sha256'],spec_unchanged=base.sha(spec_path)==manifest['spec_sha256'],no_automatic_promotion=True);base.atomic_json(directory/'summary.json',summary);print(json.dumps(dict(status=summary['status'],accepted_batches=summary['accepted_batches'],training_cameras=summary['training_transition_cameras'],cameras=summary['new_transition_cameras'],initials=summary['new_initial_cameras'],wall_seconds=summary['wall_seconds'])),flush=True)
    fresh=directory/'fresh';fresh.mkdir();drawn=select_starts(spec['fresh_root'],spec['fresh_tag'],exclusions(directory),32);seeds=drawn.pop('TRAIN');drawn.update(probe=seeds[:32],unused_reserved=seeds[32:],purpose='new greedy frozen qualification; no training/admission');base.atomic_json(fresh/'schedules.json',drawn)
    sample=[];matrices=[]
    for e in bank['bank']:
        assert base.sha(e['raw_feature'])==e['raw_feature_sha256'];X=sp.load_npz(e['raw_feature']).tocsr();indices=np.linspace(0,X.shape[0]-1,8,dtype=int);matrices.append(X[indices]);sample.append(dict(seed=e['seed'],raw=e['raw_feature'],raw_sha256=e['raw_feature_sha256'],rows=indices.tolist()))
    X=sp.vstack(matrices,format='csr');sp.save_npz(directory/'calibration-raw.npz',X);base.atomic_json(directory/'calibration-sample.json',dict(equal_episode=True,rows_per_episode=8,episodes=len(sample),sources=sample));P=current.basis.batch(X);F=np.column_stack((X.multiply(1/den).toarray(),P/phi));q0=exact_q(initial,X);q=q0.copy();initial_rms=float(np.sqrt(np.mean((q0-q0.mean(1,keepdims=True))**2)));assert initial_rms>0;temperature,actual=calibrate_temperature(q0);base.atomic_json(directory/'calibration.json',dict(initial_centered_q_rms=initial_rms,temperature=temperature,expected_nongreedy_fraction=actual,target_nongreedy_fraction=.05,rows=X.shape[0],training_only=True,temperature_frozen_for_all_batches=True,fixed_RMS_step_fraction=.01,fixed_mean_probability_TV_cap=.01));print(json.dumps(dict(phase='actor_calibrated',initial_centered_q_rms=initial_rms,temperature=temperature,expected_nongreedy=actual,elapsed=time.monotonic()-began)),flush=True)
    rng=np.random.default_rng(spec['train_seed']);phases={phase:sorted(s for s in byseed if base.leftward(s)==(phase=='left')) for phase in ('left','right')};starts={phase:rng.choice(values,32,replace=False).tolist() for phase,values in phases.items()};jobs=[[dict(batch=b,episode=i,seed=int(starts['left' if i<2 else 'right'][b]),action_seed=int(rng.integers(1,2**32,dtype=np.uint64))) for i in range(4)] for b in range(32)];assert len({j['action_seed'] for batch in jobs for j in batch})==128;base.atomic_json(directory/'training-schedule.json',dict(batches=jobs,phase_balanced=True,selection='uniform TRAIN starts per phase without outcome selection; each start two independent action seeds; before any actor games',maximum_transition_cameras=64000,maximum_batches=32))
    (directory/'training').mkdir();current_model=current_root;history=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        for b,batch in enumerate(jobs):
            remaining=64000-summary['training_transition_cameras']
            if time.monotonic()>=train_deadline-2 or remaining<4:break
            futures=[pool.submit(task,str(directory/'training'),str(current_model),job,temperature,den,phi,train_deadline-1,remaining//4) for job in batch];episodes=[f.result() for f in futures];cams=sum(e['transitions'] for e in episodes);initials=sum(e['initials'] for e in episodes);summary['training_transition_cameras']+=cams;summary['training_initials']+=initials;summary['new_transition_cameras']+=cams;summary['new_initial_cameras']+=initials;assert summary['training_transition_cameras']<=64000
            if not all(e['complete'] for e in episodes):summary['training_stop']='partial_batch_not_updated';break
            gradient=paired_gradient(episodes);delta,diagnostic=capped_step(gradient,L,F,q,temperature,initial_rms);W,R=undo(current,delta,den,phi);updated=FrozenPairwiseReadout(W,current.basis,R);newq=exact_q(updated,X);exact_change=newq-q;predicted=(F@delta.T)@CONTRAST.T;np.testing.assert_allclose(exact_change,predicted,atol=1e-10,rtol=1e-8);actual_tv=float(np.mean(np.sum(abs(probabilities(newq,temperature)-probabilities(q,temperature)),axis=1)/2));assert np.sqrt(np.mean(exact_change**2))<=.01*initial_rms+1e-10 and actual_tv<=.01+1e-10;diagnostic.update(actual_deployed_probability_TV=actual_tv,actual_deployed_function_rms=float(np.sqrt(np.mean(exact_change**2))),actual_deployed_greedy_changes=int(np.count_nonzero(np.argmax(newq,1)!=np.argmax(q,1))),cumulative_function_rms=float(np.sqrt(np.mean((newq-q0)**2))),cumulative_greedy_changes=int(np.count_nonzero(np.argmax(newq,1)!=np.argmax(q0,1))),rewards=[e['hits'] for e in episodes],within_start_reward_differences=[episodes[0]['hits']-episodes[1]['hits'],episodes[2]['hits']-episodes[3]['hits']],equal_total_pairs=sum(episodes[i]['hits']==episodes[i+1]['hits'] for i in (0,2)),nonzero_reward_to_go_pair_credits=sum(bool(np.any(episodes[i]['credit']-episodes[i+1]['hits']*episodes[i]['eligibility']+episodes[i+1]['credit']-episodes[i]['hits']*episodes[i+1]['eligibility'])) for i in (0,2)),batch=b,training_cameras=summary['training_transition_cameras']);current=updated;q=newq;summary['accepted_batches']+=1;current_model=directory/f'checkpoint_{b+1:02d}';current_model.mkdir();base.save_array(current_model/'proposal.npy',W);save_readout(current_model/'rbf-model.npz',W,current.basis,R);(current_model/'radial-cap.json').write_bytes(Path(bank['radial_cap']).read_bytes());base.atomic_json(current_model/'update.json',diagnostic);history.append(diagnostic);base.atomic_json(directory/'training-progress.json',dict(history=history,last_endpoint=str(current_model),transition_cameras=summary['training_transition_cameras'],initials=summary['training_initials']));print(json.dumps(dict(phase='actor_batch',**diagnostic,elapsed=time.monotonic()-began)),flush=True)
    summary['training_history']=history;summary['training_wall_seconds']=time.monotonic()-began
    if not history:summary['status']='no_complete_actor_batch';finish();return
    for name in ('proposal.npy','rbf-model.npz','radial-cap.json'):(directory/name).write_bytes((current_model/name).read_bytes())
    summary.update(final_model_sha256=base.sha(directory/'rbf-model.npz'),final_base_weights_sha256=base.sha(directory/'proposal.npy'),final_checkpoint=str(current_model),final_endpoint_only=True)
    def check(name,seeds,cached=None,seconds=150):
        p=directory/(name+'-bank.json');base.atomic_json(p,dict(bank,bank=[byseed[s] for s in seeds]));native_check(directory/name,p,directory/'proposal.npy',cached_native_root=cached,model_bundle=directory/'rbf-model.npz',native_seconds=seconds,practical_later_screen=True,absolute_deadline=deadline);s=base.read(directory/name/'summary.json');summary[name]=s;summary['new_transition_cameras']+=s['native']['new_transition_cameras'];summary['new_initial_cameras']+=s['native']['new_initials'];summary['status']=name+('_complete' if s['status']=='complete' else '_bounded_partial');return s
    def useful(s):return s['conditional_fresh_eligible'] and s['candidate']['first_incoming_returns']>=s['baseline']['first_incoming_returns'] and s['candidate_more']['tail']['2']>=s['baseline_more']['tail']['2']
    sentinel=check('sentinel',spec['sentinel'])
    if not useful(sentinel):finish();return
    full=check('full',list(byseed),directory/'sentinel',600);summary['full_fresh_gate']=useful(full)
    if summary['full_fresh_gate'] and time.monotonic()<deadline:
        fresh_run(fresh,current_root,directory/'full',absolute_deadline=deadline);s=base.read(fresh/'summary.json');summary.update(fresh=s,status='fresh_complete' if s['status']=='complete' else 'fresh_bounded_partial');summary['new_transition_cameras']+=s['fresh_transition_cameras'];summary['new_initial_cameras']+=s['initial_cameras']
    finish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--manifest',required=True);a=p.parse_args();run(a.directory,a.manifest)
