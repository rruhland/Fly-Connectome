"""One approved fixed512 pairwise residual comparison, one absolute600s clock."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from pong_fixed_pairwise import FixedPairwise,FrozenPairwiseReadout,save_readout
from run_rally_rbf_residual import fit_residual
from run_rally_earned_merge import load_case
from run_rally_fixed_bank_check import run as native_check
import run_practical_pong_v2 as base

SEED=831496027
DIM=512
EXPECTED='3ac7f87fbac23abab14e633b38cfe1e50986a8f03bb38b049bed8f4cc840bd5e'
FILES=('run_rally_pairwise_residual.py','pong_fixed_pairwise.py','pong_fixed_rbf.py','run_rally_rbf_residual.py','run_rally_earned_merge.py','run_rally_raw_optimizer.py','run_rally_fixed_bank_check.py','run_rally_raw_fresh_screen.py','run_rally_guarded_curriculum.py','rally_action_equivalence.py')


def array_hash(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()


def calibrate(rows,scale,sample):
    byseed={e['seed']:X for e,r,X in rows};blocks=[];weights=[];mask=[]
    assert len(sample['episodes'])==len(rows)==160 and {e['seed'] for e in sample['episodes']}==set(byseed)
    for e in sample['episodes']:
        ix=e['rows'];blocks.append(byseed[e['seed']][ix]);weights.extend([1/(160*len(ix))]*len(ix));mask.extend([e['explanatory_split']=='fit']*len(ix))
    X=sp.vstack(blocks,format='csr');assert X.shape[0]==2048;w=np.array(weights);mask=np.array(mask);Z=X.multiply(1/scale).toarray();mean=w@Z;rng=np.random.default_rng(SEED);hashes=rng.integers(DIM,size=(2,X.shape[1]));signs=rng.choice([-1,1],size=(2,X.shape[1]));raw=FixedPairwise(hashes,signs,scale,mean,np.ones(DIM));Y=raw.batch(X);rms=np.sqrt(w@(Y*Y));positive=rms[rms>0]
    if not len(positive):raise ValueError('all sketch RMS channels are zero; no model retry')
    floor=1e-6*float(np.median(positive));denominator=np.maximum(rms,floor);basis=FixedPairwise(hashes,signs,scale,mean,denominator);Y=Y/denominator
    if not np.isfinite(Y).all() or np.max(np.std(Y,axis=0))==0:raise ValueError('nonfinite or constant pairwise features; no model retry')
    A=Z[mask];B=Z[~mask];am=A.mean(0);Ac=A-am;Bc=B-am;K=Ac@Ac.T;ridge=max(1e-12,1e-6*float(np.trace(K)/len(K)));factor=la.cho_factor(K+ridge*np.eye(len(K)),lower=True,check_finite=False);cross=Bc@Ac.T
    def explain(T,V):
        mu=T.mean(0);dual=la.cho_solve(factor,T-mu,check_finite=False);return dict(training_fraction=float(np.sum((T-mu-K@dual)**2)/max(1e-30,np.sum((T-mu)**2))),validation_fraction=float(np.sum((V-mu-cross@dual)**2)/max(1e-30,np.sum((V-mu)**2))))
    control=(X@raw.projections[0]).toarray()-raw.mean_sketch[0];linear=explain(Y[mask],Y[~mask]);linear_control=explain(control[mask],control[~mask]);centered=Y-w@Y;cov=centered.T@(w[:,None]*centered);eig=la.eigvalsh(cov);energy=np.maximum(0,eig);effective=float(energy.sum()**2/(energy@energy));error=0.;start=time.monotonic()
    for i in range(X.shape[0]):
        lo,hi=X.indptr[i:i+2];error=max(error,float(np.max(np.abs(basis.features(X.indices[lo:hi],X.data[lo:hi])-Y[i]))))
    micros=(time.monotonic()-start)*1e6/X.shape[0]
    if error>1e-10:raise ValueError('sparse deployment versus fitted batch feature mismatch')
    report=dict(samples=2048,episodes=160,seed=SEED,outputs=DIM,equal_episode_mean_and_RMS=True,RMS_rule='root equal-episode mean square; floor1e-6times median positive TRAIN RMS',RMS_floor=floor,RMS_floored_channels=int(np.sum(rms<floor)),raw_RMS_quantiles=np.quantile(rms,[0,.01,.25,.5,.75,.99,1]).tolist(),denominator_quantiles=np.quantile(denominator,[0,.01,.25,.5,.75,.99,1]).tolist(),effective_centered_rank=effective,covariance_numerical_rank=int(np.sum(energy>energy.max()*1e-12)),covariance_eigenvalue_quantiles=np.quantile(energy,[0,.25,.5,.75,1]).tolist(),heldout_TRAIN_linear_explanation=linear,heldout_TRAIN_linear_control=linear_control,explanatory_fit_episodes=80,explanatory_validation_episodes=80,explanatory_ridge=ridge,maximum_batch_vs_deployed_feature_error=error,feature_microseconds=micros,diagnostic_is_not_capacity_or_policy_utility_proof=True,low_rank_is_not_functional_stop=True,new_observations=0)
    return basis,report


def run(directory,bank_path,index_path,sample_path,sentinel_schedule,prior_root,implementation_seconds):
    directory=Path(directory);bank_path=Path(bank_path);index_path=Path(index_path);sample_path=Path(sample_path);prior_root=Path(prior_root);assert not directory.exists();directory.mkdir();began=time.monotonic();deadline=began+600;base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=600,implementation_seconds_separate=implementation_seconds));bank=base.read(bank_path);index=base.read(index_path);assert base.sha(bank['current_head'])==bank['current_head_sha256']==index['reference_sha256']==EXPECTED and bank['source']==base.source_signature();W=np.load(bank['current_head']);(directory/'proposal.npy').write_bytes(Path(bank['current_head']).read_bytes());(directory/'radial-cap.json').write_bytes(Path(bank['radial_cap']).read_bytes());files={n:base.sha(Path(__file__).with_name(n)) for n in FILES};manifest=dict(source=base.source_signature(),files=files,bank=str(bank_path),bank_sha256=base.sha(bank_path),index=str(index_path),index_sha256=base.sha(index_path),saved_sample=str(sample_path),saved_sample_sha256=base.sha(sample_path),sentinel_schedule=str(sentinel_schedule),sentinel_schedule_sha256=base.sha(sentinel_schedule),reference_sha256=EXPECTED,features=DIM,seed=SEED,seconds=600,fit_seconds=60,base_frozen=True,original_transport=True,no_sweep=True,no_fresh_admission=True,legacy_bundle_filename='rbf-model.npz; content dispatches fixed-centered-tensorsketch2-v1');base.atomic_json(directory/'manifest.json',manifest);summary=dict(status='calibration_in_progress',new_transition_cameras=0,new_initial_cameras=0)
    def finish():
        summary.update(wall_seconds=time.monotonic()-began,source_frozen=manifest['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files},base_unchanged=base.sha(bank['current_head'])==EXPECTED,no_automatic_promotion=True);base.atomic_json(directory/'summary.json',summary);print(json.dumps(dict(status=summary['status'],new_transition_cameras=summary['new_transition_cameras'],new_initial_cameras=summary['new_initial_cameras'],wall_seconds=summary['wall_seconds'],source_frozen=summary['source_frozen'])),flush=True)
    rows=[(e,base.read(Path(e['receipt'])),sp.load_npz(e['raw_feature']).tocsr()) for e in bank['bank']];scale=np.load(index['feature_metric']);basis,diagnostic=calibrate(rows,scale,base.read(sample_path));base.atomic_json(directory/'basis-diagnostic.json',diagnostic);(directory/'calibration-sample.json').write_bytes(sample_path.read_bytes());save_readout(directory/'zero-rbf-model.npz',W,basis,np.zeros((3,DIM)));assets={n:array_hash(a) for n,a in dict(base=W,hashes=basis.hashes,signs=basis.signs,scale=basis.scale,mean=basis.mean,rms=basis.rms,zero_residual=np.zeros((3,DIM))).items()};base.atomic_json(directory/'asset-hashes.json',dict(assets=assets,zero_bundle_sha256=base.sha(directory/'zero-rbf-model.npz'),feature_implementation_sha256=files['pong_fixed_pairwise.py']));print(json.dumps(dict(phase='basis',diagnostic=diagnostic,elapsed=time.monotonic()-began)),flush=True)
    zero=FrozenPairwiseReadout(W,basis,np.zeros((3,DIM)));checked=0
    for e,r,X in rows:
        for i,record in enumerate(r['records']):
            lo,hi=X.indptr[i:i+2];a,q=zero.choose(X.indices[lo:hi],X.data[lo:hi]);assert a==record['action'] and np.array_equal(q,np.asarray(record['q']));checked+=1
    base.atomic_json(directory/'zero-initialization-parity.json',dict(passed=True,exact_deployed_choose_rows=checked,exact_Q_and_tie_behavior=True,new_observations=0));entry=next(e for e in index['cases'] if e['seed']==104562292);donor=load_case(entry);records=base.read(Path(entry['records']));assert donor['X'].shape[0]==848 and entry['target_hits']==5 and entry['forced_transitions']==list(range(744,754));prefix=dict(donor,X=donor['X'][:754].tocsr(),actions=donor['actions'][:754]);assert sum(r['hits'] for r in records[:754])==4 and len(prefix['interventions'])==10;cases=[prefix];G=np.zeros((DIM,DIM));episodes=0
    for e,r,X in rows:
        hits=[i for i,v in enumerate(r['records']) if v['hits']]
        if not hits:continue
        cases.append(load_case(dict(e,end=hits[0]+1,source_head=bank['current_head'],source_head_sha256=EXPECTED)))
        if e['seed']==104562292:continue
        Phi=basis.batch(X[:hits[-1]+1]);G+=Phi.T@Phi/len(Phi);episodes+=1
    assert len(cases)==144 and episodes==142;base.atomic_json(directory/'credit-evidence.json',dict(full_donor=entry,full_earned_rows=848,full_earned_hits=5,hard_rows=754,hard_prefix_hits=4,hard_prefix_is_not_fifth_return_witness=True,full_provenance_validated_before_slicing=True,opening_prefixes=143,new_observations=0));G/=episodes;normalizer=float(np.trace(G)/DIM);assert normalizer>0;G/=normalizer;G.flat[::DIM+1]+=.01;normal=base.read(base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json');R,report=fit_residual(W,basis,cases,G,normal['kappa'],min(deadline,time.monotonic()+60),FrozenPairwiseReadout);base.atomic_json(directory/'fit-report.json',report);summary.update(basis=diagnostic,fit=report);print(json.dumps(dict(phase='fit',report=report,elapsed=time.monotonic()-began)),flush=True)
    if R is None:summary['status']='no_verified_residual_fit';finish();return
    save_readout(directory/'rbf-model.npz',W,basis,R);base.save_array(directory/'residual.npy',R);base.atomic_json(directory/'score-objective.json',dict(episodes=episodes,mean_diagonal_normalizer=normalizer,ridge_fraction=.01,normalized_objective=float(np.sum((R@G)*R)),new_observations=0));assets.update(residual=array_hash(R));base.atomic_json(directory/'asset-hashes.json',dict(assets=assets,zero_bundle_sha256=base.sha(directory/'zero-rbf-model.npz'),model_bundle_sha256=base.sha(directory/'rbf-model.npz'),feature_implementation_sha256=files['pong_fixed_pairwise.py']));readout=FrozenPairwiseReadout(W,basis,R);timing={}
    for label,p in [('base',readout.base),('residual',readout)]:
        start=time.monotonic()
        for i in range(donor['X'].shape[0]):
            lo,hi=donor['X'].indptr[i:i+2];p.choose(donor['X'].indices[lo:hi],donor['X'].data[lo:hi])
        timing[label+'_choose_microseconds']=(time.monotonic()-start)*1e6/donor['X'].shape[0]
    base.atomic_json(directory/'inference-timing.json',dict(timing,rows=848,new_observations=0))
    def check(name,entries,cached=None,seconds=120):
        path=directory/(name+'-bank.json');base.atomic_json(path,dict(bank,bank=entries));native_check(directory/name,path,directory/'proposal.npy',cached_native_root=cached,alternate_native_root=prior_root,model_bundle=directory/'rbf-model.npz',native_seconds=seconds,practical_later_screen=True,absolute_deadline=deadline);s=base.read(directory/name/'summary.json');summary[name]=s;summary['new_transition_cameras']+=s['native']['new_transition_cameras'];summary['new_initial_cameras']+=s['native']['new_initials'];summary['status']=name+('_complete' if s['status']=='complete' else '_bounded_partial');return s
    if time.monotonic()>=deadline:summary['status']='budget_exhausted_before_target';finish();return
    target=check('target',[e for e in bank['bank'] if e['seed']==104562292],seconds=30);summary['actual_target_fifth_return']=target['status']=='complete' and target['candidate']['own_returns']>=5
    if not summary['actual_target_fifth_return']:summary['status']='rejected_no_actual_fifth_target_return' if target['status']=='complete' else 'target_bounded_partial';finish();return
    schedule=base.read(sentinel_schedule);base.atomic_json(directory/'sentinel-schedule.json',schedule);entries=[e for s in schedule['seeds'] for e in bank['bank'] if e['seed']==s];assert len(entries)==32
    if time.monotonic()>=deadline:summary['status']='budget_exhausted_after_target';finish();return
    s=check('sentinel',entries,directory/'target');advance=s['conditional_fresh_eligible'] and s['candidate']['first_incoming_returns']>=s['baseline']['first_incoming_returns'];summary['advance_to_full']=advance
    if not advance or time.monotonic()>=deadline:finish();return
    whole=check('full',bank['bank'],directory/'sentinel',500)
    if whole['conditional_fresh_eligible'] and time.monotonic()<deadline:
        import run_rally_baseline_pong as schedules
        from run_rally_raw_fresh_screen import run as fresh_run
        previous=set();prior_schedules={}
        def collect(value):
            if type(value) is int and 0<value<2**32:previous.add(value%base.MODULUS)
            elif isinstance(value,list):
                for item in value:collect(item)
            elif isinstance(value,dict):
                for item in value.values():collect(item)
        for path in sorted(base.EXPERIMENTS.rglob('schedules.json')):collect(base.read(path));prior_schedules[str(path)]=base.sha(path)
        fresh=directory/'fresh';fresh.mkdir();schedules.ROOT=1943906127;schedules.TAG=62945;drawn=schedules.schedule();seeds=drawn.pop('train');assert not {s%base.MODULUS for s in seeds}&previous and not set(seeds)&{e['seed'] for e in bank['bank']};drawn.update(probe=seeds[:32],unused_reserved=seeds[32:],excluded_prior_schedule_sha256=prior_schedules,selection='genuinely unused unfiltered paired native games; no fitting/no admission');base.atomic_json(fresh/'schedules.json',drawn);fresh_run(fresh,Path(bank['current_head']).parent,directory/'full',absolute_deadline=deadline);new=base.read(fresh/'summary.json');summary.update(status='fresh_complete' if new['status']=='complete' else 'fresh_bounded_partial',fresh=new);summary['new_transition_cameras']+=new['fresh_transition_cameras'];summary['new_initial_cameras']+=new['initial_cameras']
    finish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--bank',required=True);p.add_argument('--index',required=True);p.add_argument('--sample',required=True);p.add_argument('--sentinel-schedule',required=True);p.add_argument('--prior-root',required=True);p.add_argument('--implementation-seconds',type=float,required=True);a=p.parse_args();run(a.directory,a.bank,a.index,a.sample,a.sentinel_schedule,a.prior_root,a.implementation_seconds)
