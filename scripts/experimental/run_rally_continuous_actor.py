"""Approved single600s continuous visual residual pilot; no model/map sweep."""
import argparse,json,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from pong_continuous_actor import task
from pong_continuous_visual import fit_basis,ContinuousReadout,save_readout
from pong_fixed_rbf import load_readout
from pong_full_rally_actor import paired_gradient,capped_step,probabilities
from pong_constrained_residual import CONTRAST
from run_rally_fixed_bank_check import run as native_check
from run_rally_raw_fresh_screen import run as fresh_run
from run_rally_repertoire_collection import exclusions,select_starts
import run_practical_pong_v2 as base

FILES=('run_rally_continuous_actor.py','pong_continuous_actor.py','pong_continuous_visual.py','pong_full_rally_actor.py','pong_fixed_rbf.py','pong_fixed_pairwise.py','run_rally_fixed_bank_check.py','run_rally_guarded_curriculum.py','rally_action_equivalence.py','run_rally_raw_optimizer.py','run_rally_raw_fresh_screen.py','run_rally_repertoire_collection.py','rally_raw_capture.py')


def run(directory,spec_path):
    directory=Path(directory);spec_path=Path(spec_path);assert not directory.exists();directory.mkdir();began=time.monotonic();deadline=began+600;train_deadline=began+300;spec=base.read(spec_path);base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,training_deadline_monotonic=train_deadline,seconds=600,implementation_seconds_separate_estimate=spec['implementation_seconds_separate_estimate']));(directory/'spec.json').write_bytes(spec_path.read_bytes());source=base.source_signature();files={n:base.sha(Path(__file__).with_name(n)) for n in FILES};bank=base.read(spec['bank']);assert base.sha(spec['bank'])==spec['bank_sha256'] and bank['source']==source
    for p,digest in spec['pinned_assets'].items():assert base.sha(p)==digest
    original=load_readout(bank['current_model_bundle']);original_root=directory/'original';original_root.mkdir()
    for src,name in [(bank['current_head'],'proposal.npy'),(bank['current_model_bundle'],'rbf-model.npz'),(bank['radial_cap'],'radial-cap.json')]:(original_root/name).write_bytes(Path(src).read_bytes())
    manifest=dict(source=source,files=files,spec_sha256=base.sha(spec_path),approved_major_change=True,normalization_floor_fraction=.1,initial_policy_sha256=bank['current_policy_sha256'],initial_base_weights_sha256=bank['current_head_sha256'],new_output_features=256,new_output_parameters=768,contrast_coordinates=512,base_W_R_vision_frozen=True,raw_only_certificates_disabled_for_nonzero_branch=True,no_backprop=True,no_sweep=True,endpoint='FINAL completed batch only');base.atomic_json(directory/'manifest.json',manifest);summary=dict(status='calibrating',new_transition_cameras=0,new_initial_cameras=0,calibration_transition_cameras=0,calibration_initials=0,training_transition_cameras=0,training_initials=0,accepted_batches=0)
    def finish():
        summary.update(wall_seconds=time.monotonic()-began,source_frozen=source==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in FILES},assets_unchanged=all(base.sha(p)==h for p,h in spec['pinned_assets'].items()),bank_unchanged=base.sha(spec['bank'])==spec['bank_sha256'],spec_unchanged=base.sha(spec_path)==manifest['spec_sha256'],no_automatic_promotion=True);base.atomic_json(directory/'summary.json',summary);print(json.dumps(dict(status=summary['status'],batches=summary['accepted_batches'],calibration_cameras=summary['calibration_transition_cameras'],training_cameras=summary['training_transition_cameras'],cameras=summary['new_transition_cameras'],initials=summary['new_initial_cameras'],elapsed=summary['wall_seconds'])),flush=True)
    fresh=directory/'fresh';fresh.mkdir();drawn=select_starts(spec['fresh_root'],spec['fresh_tag'],exclusions(directory),32);seeds=drawn.pop('TRAIN');drawn.update(probe=seeds[:32],unused_reserved=seeds[32:],purpose='new paired greedy qualification; no admission');base.atomic_json(fresh/'schedules.json',drawn);calib_dir=directory/'calibration';calib_dir.mkdir();history=[];temperature=.008161197682294006;initial_rms=.800332956301049
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        futures=[pool.submit(task,str(calib_dir),str(original_root),dict(seed=s,tag=f'calibration_{i:02d}'),min(train_deadline,time.monotonic()+90),256) for i,s in enumerate(spec['calibration_seeds'])];calibration=[f.result() for f in futures];summary['calibration_transition_cameras']=sum(r['transitions'] for r in calibration);summary['calibration_initials']=sum(r['initials'] for r in calibration);summary['new_transition_cameras']+=summary['calibration_transition_cameras'];summary['new_initial_cameras']+=summary['calibration_initials'];assert summary['calibration_transition_cameras']<=2048
        if not all(r['allocation_complete'] and r['transitions']>0 for r in calibration):summary['status']='calibration_budget_partial';finish();return
        episodes=[r['scenes'] for r in calibration];scenes=[s for e in episodes for s in e];q0=np.concatenate([r['q'] for r in calibration]);X=sp.vstack([sp.load_npz(r['raw_feature_file']) for r in calibration],format='csr')
        try:basis,F,G,weights,report=fit_basis(episodes)
        except (AssertionError,ValueError,FloatingPointError) as error:summary.update(status='calibration_invalid_no_map_retry',calibration_error=repr(error));finish();return
        start=time.monotonic();[basis.features(s) for s in scenes[:100]];report['mean_feature_microseconds_first100']=1e6*(time.monotonic()-start)/min(100,len(scenes));report['elapsed_since_experiment_start']=time.monotonic()-began;base.atomic_json(directory/'calibration-report.json',report);base.save_array(directory/'score-metric.npy',G);L=la.cholesky(G,lower=True,check_finite=False);current=ContinuousReadout(original,basis,np.zeros((3,256)));zero_errors=[]
        for i,scene in enumerate(scenes):
            lo,hi=X.indptr[i:i+2];ix=X.indices[lo:hi];x=X.data[lo:hi];a,q=current.choose_features(ix,x,basis.features(scene));assert np.array_equal(q,q0[i]) and a==int(np.argmax(q0[i]));np.testing.assert_allclose(basis.features(scene),F[i],atol=1e-14)
        base.atomic_json(directory/'zero-output-parity.json',dict(passed=True,actual_calibration_rows=len(scenes),literal_original_sparse_Q_and_choose=True,raw_only_reuse_not_used=True));print(json.dumps(dict(phase='continuous_calibrated',report=report,elapsed=time.monotonic()-began)),flush=True);(directory/'training').mkdir();coefficients=np.zeros((2,256));q=q0.copy();current_model=directory/'checkpoint_00';current_model.mkdir();save_readout(current_model/'rbf-model.npz',current)
        for src,name in [(bank['current_head'],'proposal.npy'),(bank['radial_cap'],'radial-cap.json')]:(current_model/name).write_bytes(Path(src).read_bytes())
        for b,batch in enumerate(spec['training_batches']):
            remaining=48000-summary['training_transition_cameras']
            if time.monotonic()>=train_deadline-2 or remaining<4:break
            futures=[pool.submit(task,str(directory/'training'),str(current_model),job,train_deadline-1,remaining//4,temperature) for job in batch];rs=[f.result() for f in futures];summary['training_transition_cameras']+=sum(r['transitions'] for r in rs);summary['training_initials']+=sum(r['initials'] for r in rs);summary['new_transition_cameras']+=sum(r['transitions'] for r in rs);summary['new_initial_cameras']+=sum(r['initials'] for r in rs);assert summary['training_transition_cameras']<=48000
            if not all(r['complete'] for r in rs):summary['training_stop']='partial_batch_paid_not_updated';break
            gradient=paired_gradient(rs);delta,diagnostic=capped_step(gradient,L,F,q,temperature,initial_rms);coefficients+=delta;current=ContinuousReadout(original,basis,CONTRAST@coefficients);newq=np.array([q0[i]+current.output@f for i,f in enumerate(F)]);change=newq-q;np.testing.assert_allclose(change,(F@delta.T)@CONTRAST.T,atol=1e-10,rtol=1e-8);tv=float(np.mean(np.sum(abs(probabilities(newq,temperature)-probabilities(q,temperature)),axis=1)/2));assert np.sqrt(np.mean(change**2))<=.01*initial_rms+1e-10 and tv<=.01+1e-10;diagnostic.update(batch=b,rewards=[r['hits'] for r in rs],actual_deployed_RMS=float(np.sqrt(np.mean(change**2))),actual_deployed_TV=tv,cumulative_RMS=float(np.sqrt(np.mean((newq-q0)**2))),cumulative_greedy_changes=int(np.count_nonzero(np.argmax(newq,1)!=np.argmax(q0,1))),training_cameras=summary['training_transition_cameras']);q=newq;history.append(diagnostic);summary['accepted_batches']+=1;current_model=directory/f'checkpoint_{b+1:02d}';current_model.mkdir();save_readout(current_model/'rbf-model.npz',current)
            for src,name in [(bank['current_head'],'proposal.npy'),(bank['radial_cap'],'radial-cap.json')]:(current_model/name).write_bytes(Path(src).read_bytes())
            base.atomic_json(current_model/'update.json',diagnostic);base.atomic_json(directory/'training-progress.json',dict(history=history,last_endpoint=str(current_model),transition_cameras=summary['training_transition_cameras'],initials=summary['training_initials']));print(json.dumps(dict(phase='continuous_batch',**diagnostic,elapsed=time.monotonic()-began)),flush=True)
    summary.update(training_history=history,training_wall_seconds=time.monotonic()-began)
    if not history:summary['status']='no_complete_continuous_batch';finish();return
    for name in ('proposal.npy','rbf-model.npz','radial-cap.json'):(directory/name).write_bytes((current_model/name).read_bytes())
    summary.update(final_model_sha256=base.sha(directory/'rbf-model.npz'),final_base_weights_sha256=base.sha(directory/'proposal.npy'),final_endpoint=str(current_model));entries={e['seed']:e for e in bank['bank']};screen_bank=directory/'screen-bank.json';base.atomic_json(screen_bank,dict(bank,bank=[entries[s] for s in spec['screen']]));native_check(directory/'screen32',screen_bank,directory/'proposal.npy',model_bundle=directory/'rbf-model.npz',practical_later_screen=True,native_seconds=max(1,deadline-time.monotonic()),absolute_deadline=deadline);s=base.read(directory/'screen32/summary.json');summary.update(screen32=s,status='screen32_complete' if s['status']=='complete' else 'screen32_bounded_partial');summary['new_transition_cameras']+=s['native']['new_transition_cameras'];summary['new_initial_cameras']+=s['native']['new_initials'];qualify=s['conditional_fresh_eligible'] and s['candidate']['first_incoming_returns']>=s['baseline']['first_incoming_returns'] and s['candidate_more']['tail']['2']>=s['baseline_more']['tail']['2'] and s['candidate_more']['later_fraction']>=s['baseline_more']['later_fraction'];summary['fresh_gate']=qualify
    if qualify and time.monotonic()<deadline:
        fresh_run(fresh,Path(bank['current_head']).parent,directory/'screen32',absolute_deadline=deadline);new=base.read(fresh/'summary.json');summary.update(fresh=new,status='fresh_complete' if new['status']=='complete' else 'fresh_bounded_partial');summary['new_transition_cameras']+=new['fresh_transition_cameras'];summary['new_initial_cameras']+=new['initial_cameras']
    finish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--manifest',required=True);a=p.parse_args();run(a.directory,a.manifest)
