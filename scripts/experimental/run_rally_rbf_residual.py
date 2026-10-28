"""One approved, clock-bounded fixed RBF residual comparison; no sweeps."""
import argparse,json,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import osqp
from fly_connectome.control.policy import FrozenLinearPolicy
from fly_connectome.vision.interface import VisualStateEncoder
from pong_constrained_residual import CONTRAST
from pong_fixed_rbf import FixedRBF,FrozenRBFReadout,save_readout
from run_rally_earned_merge import load_case
from run_rally_fixed_bank_check import run as native_check
import run_practical_pong_v2 as base

SAMPLE_SEED=831496027


def calibrate(rows,scale):
    rng=np.random.default_rng(SAMPLE_SEED);order=rng.permutation(len(rows));extra=set(order[:2048%len(rows)]);samples=[];provenance=[];fit_episodes=set(order[:len(rows)//2]);is_fit=[]
    for j,(entry,receipt,X) in enumerate(rows):
        n=2048//len(rows)+int(j in extra);indices=np.sort(rng.choice(X.shape[0],n,replace=False));samples.append(X[indices]);provenance.append(dict(seed=entry['seed'],rows=indices.tolist(),explanatory_split='fit' if j in fit_episodes else 'validation'));is_fit.extend([j in fit_episodes]*n)
    X=sp.vstack(samples,format='csr');Z=X.multiply(1/scale).toarray();norm=np.einsum('ij,ij->i',Z,Z);selected=[int(rng.integers(len(Z)))];distance=np.full(len(Z),np.inf)
    while len(selected)<64:
        c=Z[selected[-1]];distance=np.minimum(distance,np.maximum(0.,norm+c@c-2*(Z@c)));distance[selected]=-1.;i=int(np.argmax(distance))
        if distance[i]<=0:raise ValueError('fewer than64 distinct calibration inputs')
        selected.append(i)
    centers=Z[selected];d2=np.maximum(0.,np.einsum('ij,ij->i',centers,centers)[:,None]+np.einsum('ij,ij->i',centers,centers)[None,:]-2*(centers@centers.T));np.fill_diagonal(d2,np.inf);sigma=float(np.median(np.sqrt(np.sort(d2,axis=1)[:,:8])));basis=FixedRBF(centers,scale,sigma);Y=basis.batch(X);mask=np.asarray(is_fit);A=Z[mask];B=Z[~mask];a_mean=A.mean(0);Ac=A-a_mean;Bc=B-a_mean;ym=Y[mask].mean(0);kernel=Ac@Ac.T;ridge=max(1e-12,1e-6*float(np.trace(kernel)/len(kernel)));factor=la.cho_factor(kernel+ridge*np.eye(len(kernel)),lower=True,check_finite=False)
    mean_phi=np.exp(-np.sum((centers-a_mean)**2,axis=1)/(2*sigma*sigma));J=-(a_mean-centers)*mean_phi[:,None]/(sigma*sigma);tangent_A=Ac@J.T+mean_phi;tangent_B=Bc@J.T+mean_phi
    def unexplained(train,valid):
        mean=train.mean(0);dual=la.cho_solve(factor,train-mean,check_finite=False);prediction=mean+(Bc@Ac.T)@dual;train_prediction=mean+kernel@dual;den=max(1e-30,float(np.sum((valid-mean)**2)));return dict(validation_fraction=float(np.sum((valid-prediction)**2)/den),training_fraction=float(np.sum((train-train_prediction)**2)/max(1e-30,float(np.sum((train-mean)**2)))))
    linear=unexplained(Y[mask],Y[~mask]);control=unexplained(tangent_A,tangent_B);singular=la.svdvals(Y-Y.mean(0));energy=singular**2;effective=float(energy.sum()**2/(energy@energy));tangent_all=(Z-a_mean)@J.T+mean_phi;curvature=float(np.sum((Y-tangent_all)**2)/max(1e-30,np.sum((Y-Y.mean(0))**2)));saturation=float(np.mean((Y<1e-6)|(Y>1-1e-6)));usable=effective>=4 and linear['validation_fraction']>=.05 and curvature>=.05 and saturation<.95
    pairs=rng.integers(0,len(Z),size=(512,2));difference=Z[pairs[:,0]]-Z[pairs[:,1]];total=max(1e-30,float(np.sum(difference*difference)));blocks={name:float(np.sum(difference[:,lo:hi]**2)/total) for name,(lo,hi) in VisualStateEncoder(64,64,.02).layout.items()};error=0.;start=time.monotonic()
    for i in range(len(X.indptr)-1):
        lo,hi=X.indptr[i:i+2];actual=basis.features(X.indices[lo:hi],X.data[lo:hi]);error=max(error,float(np.max(np.abs(actual-Y[i]))))
    feature_seconds=(time.monotonic()-start)/X.shape[0]
    report=dict(samples=2048,episodes=len(rows),centers=64,seed=SAMPLE_SEED,bandwidth=sigma,bandwidth_rule='median Euclidean distance across each center eight nearest other centers; fixed once',selected_sample_rows=selected,activation_quantiles=np.quantile(Y,[0,.01,.1,.5,.9,.99,1]).tolist(),per_feature_std_quantiles=np.quantile(Y.std(0),[0,.25,.5,.75,1]).tolist(),saturation_fraction=saturation,effective_centered_rank=effective,heldout_TRAIN_linear_explanation=linear,heldout_TRAIN_tangent_linear_control=control,tangent_curvature_fraction=curvature,explanatory_fit_episodes=len(fit_episodes),explanatory_validation_episodes=len(rows)-len(fit_episodes),explanatory_ridge=ridge,distance_fraction_by_existing_block=blocks,maximum_batch_vs_deployed_feature_error=error,feature_microseconds=feature_seconds*1e6,usable=usable,predeclared_diagnostic_gate='centered effective rank>=4, heldout linear unexplained>=5%, tangent curvature>=5%, less95%saturated; diagnostics do not establish policy utility',new_observations=0)
    return basis,report,provenance


def fit_residual(W,basis,cases,G,kappa,deadline,readout_class=FrozenRBFReadout):
    began=time.monotonic();X=sp.vstack([c['X'] for c in cases],format='csr');actions=np.concatenate([c['actions'] for c in cases]);required=[];qbase=[]
    for case in cases:
        assert np.array_equal(W,case['source_weights']), 'residual constraints require the exact frozen deployed base head'
        policy=FrozenLinearPolicy(case['source_weights']);forced={v['row']:v for v in case.get('interventions',[])}
        for i,a in enumerate(case['actions']):
            lo,hi=case['X'].indptr[i:i+2];choice,q=policy.choose(case['X'].indices[lo:hi],case['X'].data[lo:hi]);qbase.append(q)
            if i in forced:
                v=forced[i];assert v['reward_verified'] and v['action']==a and v['source_action']==choice;required.append([kappa*1e-4]*2)
            else:
                assert choice==a;required.append([min(kappa*1e-4,.5*max(q[a]-q[b],0.)) for b in range(3) if b!=a])
    m=basis.output_dim;assert G.shape==(m,m)
    required=np.asarray(required);qbase=np.asarray(qbase);other=np.array([[b for b in range(3) if b!=a] for a in actions]);row=np.arange(len(actions));old=qbase[row,actions,None]-qbase[row[:,None],other];contrast=CONTRAST[actions,None,:]-CONTRAST[other];Phi=basis.batch(X);values=np.repeat(Phi,2,axis=0);direction=contrast.reshape(-1,2);A=np.hstack((values*direction[:,0,None],values*direction[:,1,None]));lower=(required-old).reshape(-1);norm=np.linalg.norm(A,axis=1);assert (norm>0).all();An=A/norm[:,None];lowern=lower/norm;P=sp.block_diag((2*sp.csc_matrix(G),2*sp.csc_matrix(G)),format='csc');working=np.flatnonzero(lower>0).tolist();v=np.zeros(2*m);duals={};history=[];candidate=None;status='bounded_no_verified_residual_fit';tight=False;infeasibility=None
    for iteration in range(30):
        remaining=deadline-time.monotonic()
        if remaining<=0:break
        eps=1e-11 if tight else 1e-8;solver=osqp.OSQP();solver.setup(P=P,q=np.zeros(2*m),A=sp.csc_matrix(An[working]),l=lowern[working],u=np.full(len(working),np.inf),verbose=False,polishing=True,eps_abs=eps,eps_rel=eps,max_iter=30000,time_limit=max(.01,min(remaining,12.)));solver.warm_start(x=v,y=np.array([duals.get(i,0.) for i in working]));answer=solver.solve(raise_error=False)
        if 'infeasible' in answer.info.status:
            status='solver_reported_'+answer.info.status
            if answer.prim_inf_cert is not None:
                lam=-np.asarray(answer.prim_inf_cert);infeasibility=dict(working_constraints=len(working),minimum_multiplier=float(lam.min()),normalized_stationarity_max=float(np.max(np.abs(An[working].T@lam))),positive_lower_combination=float(lowern[working]@lam))
            break
        if answer.x is None or not np.isfinite(answer.x).all():status='nonfinite_no_fallback';break
        v=np.asarray(answer.x)
        if answer.y is not None and np.isfinite(answer.y).all():duals={i:float(y) for i,y in zip(working,answer.y)}
        violation=lower-A@v;maximum=max(0.,float(violation.max()));history.append(dict(round=iteration+1,working=len(working),maximum_original_logit_violation=maximum,solver_status=answer.info.status,iterations=int(answer.info.iter)))
        if maximum<=1e-9:
            R=CONTRAST@v.reshape(2,m);readout=readout_class(W,basis,R);wrong=[];actual_violation=0.
            for i,a in enumerate(actions):
                lo,hi=X.indptr[i:i+2];got,q=readout.choose(X.indices[lo:hi],X.data[lo:hi]);actual_violation=max(actual_violation,float(np.max(required[i]-np.array([q[a]-q[b] for b in range(3) if b!=a]))))
                if got!=a:wrong.append(i)
            if not wrong and actual_violation<=1e-9:candidate=R;status='verified_all_original_residual_choices';break
            if not tight:tight=True;continue
            status='deployed_original_certificate_failed';break
        existing=set(working);fresh=np.array([i for i in np.flatnonzero(violation>1e-9) if int(i) not in existing],dtype=int)
        if not len(fresh):
            if not tight:tight=True;continue
            status='bounded_numerical_residual_no_fallback';break
        working=sorted(existing|set(map(int,fresh[np.argsort(-violation[fresh],kind='stable')[:256]])))
    return candidate,dict(status=status,verified=candidate is not None,hard_cases=len(cases),original_rows=X.shape[0],original_inequalities=len(lower),working_inequalities=len(working),history=history,wall_seconds=time.monotonic()-began,infeasibility_diagnostic=infeasibility,objective='minimum equal-episode empirical residual score change, mean-diagonal normalized covariance plus1%ridge',residual_parameters=3*m,contrast_parameters=2*m,base_head_frozen=True,original_deployed_choices_certified=X.shape[0] if candidate is not None else 0,maximum_actual_logit_violation=actual_violation if candidate is not None else None,no_confidence_growth_objective=True,numerical_fallback=False,new_observations=0)


def run(directory,bank_path,index_path,sentinel_schedule,resume_saved_basis=False):
    directory=Path(directory);bank_path=Path(bank_path);index_path=Path(index_path)
    if resume_saved_basis:
        initial=base.read(directory/'manifest.json');clock=base.read(directory/'work-clock.json');began=clock['started_monotonic'];deadline=clock['deadline_monotonic'];assert time.monotonic()<deadline and not (directory/'fit-report.json').exists() and not (directory/'continuation-manifest.json').exists();assert initial['bank_sha256']==base.sha(bank_path) and initial['index_sha256']==base.sha(index_path);assert base.read(directory/'summary.json')['status']=='stopped_unavailable_useful_nonlinear_variation';(directory/'screen-stop-summary.json').write_bytes((directory/'summary.json').read_bytes())
    else:
        assert not directory.exists();directory.mkdir();began=time.monotonic();deadline=began+600;base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=600))
    bank=base.read(bank_path);index=base.read(index_path);W=np.load(bank['current_head']);expected='3ac7f87fbac23abab14e633b38cfe1e50986a8f03bb38b049bed8f4cc840bd5e';assert base.sha(bank['current_head'])==bank['current_head_sha256']==expected;assert bank['source']==base.source_signature();(directory/'proposal.npy').write_bytes(Path(bank['current_head']).read_bytes());(directory/'radial-cap.json').write_bytes(Path(bank['radial_cap']).read_bytes());files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_rbf_residual.py','pong_fixed_rbf.py','run_rally_raw_optimizer.py','rally_action_equivalence.py','run_rally_guarded_curriculum.py','run_rally_fixed_bank_check.py','run_rally_raw_fresh_screen.py')};manifest=dict(source=base.source_signature(),files=files,bank=str(bank_path),bank_sha256=base.sha(bank_path),index=str(index_path),index_sha256=base.sha(index_path),reference_sha256=expected,approved_features=64,base_frozen=True,vision_scaler_actuator_unchanged=True,seconds=600,fit_seconds=60,no_bandwidth_sweep=True,no_fresh_admission=True)
    if resume_saved_basis:
        manifest.update(parent_diagnostic_gate_amendment='effective rank is a heuristic, not evidence low-variance directions cannot realize correction; authorize exactly one output fit on saved basis, unchanged objective/data and original deadline',original_clock_sha256=base.sha(directory/'work-clock.json'),original_deadline_monotonic=deadline,saved_basis_sha256=base.sha(directory/'basis.npz'),preserved_screen_summary_sha256=base.sha(directory/'screen-stop-summary.json'),sentinel_schedule=str(sentinel_schedule),sentinel_schedule_sha256=base.sha(sentinel_schedule));base.atomic_json(directory/'continuation-manifest.json',manifest)
    else:base.atomic_json(directory/'manifest.json',manifest)
    rows=[(e,base.read(Path(e['receipt'])),sp.load_npz(e['raw_feature']).tocsr()) for e in bank['bank']];scale=np.load(index['feature_metric'])
    if resume_saved_basis:
        with np.load(directory/'basis.npz') as f:basis=FixedRBF(f['centers'],f['scale'],float(f['sigma']))
        assert np.array_equal(scale,basis.scale);diagnostic=base.read(directory/'basis-diagnostic.json');assert diagnostic['bandwidth']==basis.sigma and base.read(directory/'zero-initialization-parity.json')['passed']
    else:
        basis,diagnostic,sample=calibrate(rows,scale);base.atomic_json(directory/'basis-diagnostic.json',diagnostic);base.atomic_json(directory/'calibration-sample.json',dict(episodes=sample));np.savez(directory/'basis.npz',centers=basis.centers,scale=basis.scale,sigma=basis.sigma);print(json.dumps(dict(phase='basis',diagnostic=diagnostic,elapsed=time.monotonic()-began)),flush=True)
    zero=FrozenRBFReadout(W,basis,np.zeros((3,64)));checked=0
    for entry,r,X in rows:
        for i,record in enumerate(r['records']):
            lo,hi=X.indptr[i:i+2];a,q=zero.choose(X.indices[lo:hi],X.data[lo:hi]);assert a==record['action'] and np.array_equal(q,np.asarray(record['q']));checked+=1
    base.atomic_json(directory/'zero-initialization-parity.json',dict(passed=True,exact_deployed_choose_rows=checked,exact_Q_and_tie_behavior=True,new_observations=0));save_readout(directory/'zero-rbf-model.npz',W,basis,np.zeros((3,64)))
    if not diagnostic['usable'] and not resume_saved_basis:base.atomic_json(directory/'summary.json',dict(status='stopped_unavailable_useful_nonlinear_variation',basis=diagnostic,new_observations=0,wall_seconds=time.monotonic()-began,source_frozen=manifest['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files},base_unchanged=base.sha(bank['current_head'])==expected));return
    donor_entry=next(e for e in index['cases'] if e['seed']==104562292);donor=load_case(donor_entry);cases=[donor];G=np.zeros((64,64));episodes=0
    for e,r,X in rows:
        hits=[i for i,v in enumerate(r['records']) if v['hits']]
        if not hits:continue
        first=dict(e,end=hits[0]+1,source_head=bank['current_head'],source_head_sha256=expected);cases.append(load_case(first))
        if e['seed']==104562292:continue
        Phi=basis.batch(X[:hits[-1]+1]);G+=Phi.T@Phi/len(Phi);episodes+=1
    assert len(cases)==144 and episodes==142;G/=episodes;normalizer=float(np.trace(G)/64);G/=normalizer;G.flat[::65]+=.01;normal=base.read(base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json');fit_start=time.monotonic();R,report=fit_residual(W,basis,cases,G,normal['kappa'],min(deadline,fit_start+60));base.atomic_json(directory/'fit-report.json',report);print(json.dumps(dict(phase='fit',report=report,elapsed=time.monotonic()-began)),flush=True)
    if R is None:base.atomic_json(directory/'summary.json',dict(status='no_verified_residual_fit',basis=diagnostic,fit=report,new_observations=0,wall_seconds=time.monotonic()-began,source_frozen=manifest['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files},base_unchanged=base.sha(bank['current_head'])==expected));return
    readout=FrozenRBFReadout(W,basis,R);timing={}
    for name,policy in [('base',FrozenLinearPolicy(W)),('residual',readout)]:
        start=time.monotonic()
        for i in range(donor['X'].shape[0]):
            lo,hi=donor['X'].indptr[i:i+2];policy.choose(donor['X'].indices[lo:hi],donor['X'].data[lo:hi])
        timing[name+'_choose_microseconds']=(time.monotonic()-start)*1e6/donor['X'].shape[0]
    base.atomic_json(directory/'inference-timing.json',dict(timing,rows=donor['X'].shape[0],new_observations=0));save_readout(directory/'rbf-model.npz',W,basis,R);base.save_array(directory/'residual.npy',R);base.atomic_json(directory/'score-objective.json',dict(episodes=episodes,mean_diagonal_normalizer=normalizer,ridge_fraction=.01,normalized_objective=float(np.sum((R@G)*R)),new_observations=0));schedule=base.read(Path(sentinel_schedule));base.atomic_json(directory/'sentinel-schedule.json',schedule);entries=[e for s in schedule['seeds'] for e in bank['bank'] if e['seed']==s];assert len(entries)==32;sb=dict(bank,bank=entries);base.atomic_json(directory/'sentinel-bank.json',sb);native_check(directory/'sentinel',directory/'sentinel-bank.json',directory/'proposal.npy',model_bundle=directory/'rbf-model.npz',native_seconds=120,absolute_deadline=deadline);s=base.read(directory/'sentinel/summary.json');advance=s['status']=='complete' and s['candidate']['own_returns']>s['baseline']['own_returns'];summary=dict(status='sentinel_complete' if s['status']=='complete' else 'sentinel_bounded_partial',basis=diagnostic,fit=report,sentinel=s,advance_to_full=advance,wall_seconds=time.monotonic()-began);base.atomic_json(directory/'summary.json',summary)
    if advance and time.monotonic()<deadline:
        native_check(directory/'full',bank_path,directory/'proposal.npy',cached_native_root=directory/'sentinel',model_bundle=directory/'rbf-model.npz',native_seconds=500,practical_later_screen=True,absolute_deadline=deadline);full=base.read(directory/'full/summary.json');summary.update(status='full_complete' if full['status']=='complete' else 'full_bounded_partial',full=full)
        if full['conditional_fresh_eligible'] and time.monotonic()<deadline:
            import run_rally_baseline_pong as schedules
            from run_rally_raw_fresh_screen import run as fresh_run
            prior=set();prior_files={}
            def collect(value):
                if type(value) is int and 0<value<2**32:prior.add(value%base.MODULUS)
                elif isinstance(value,list):
                    for v in value:collect(v)
                elif isinstance(value,dict):
                    for v in value.values():collect(v)
            for p in sorted(base.EXPERIMENTS.rglob('schedules.json')):collect(base.read(p));prior_files[str(p)]=base.sha(p)
            fresh=directory/'fresh';fresh.mkdir();schedules.ROOT=1943906127;schedules.TAG=62913;drawn=schedules.schedule();seeds=drawn.pop('train');assert not set(seeds)&{e['seed'] for e in bank['bank']};assert not {s%base.MODULUS for s in seeds}&prior;drawn.update(probe=seeds[:32],unused_reserved=seeds[32:],selection='fresh unfiltered paired native starts; no fitting/no admission',excluded_prior_schedule_sha256=prior_files);base.atomic_json(fresh/'schedules.json',drawn);fresh_run(fresh,Path(bank['current_head']).parent,directory/'full',absolute_deadline=deadline);fresh_summary=base.read(fresh/'summary.json');summary.update(status='fresh_complete' if fresh_summary['status']=='complete' else 'fresh_bounded_partial',fresh=fresh_summary)
    summary.update(wall_seconds=time.monotonic()-began,source_frozen=manifest['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files},base_unchanged=base.sha(bank['current_head'])==expected,no_automatic_promotion=True);base.atomic_json(directory/'summary.json',summary);print(json.dumps(dict(status=summary['status'],wall_seconds=summary['wall_seconds'],source_frozen=summary['source_frozen'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--bank',required=True);p.add_argument('--index',required=True);p.add_argument('--sentinel-schedule',required=True);p.add_argument('--resume-saved-basis',action='store_true');a=p.parse_args();run(a.directory,a.bank,a.index,a.sentinel_schedule,a.resume_saved_basis)
