"""One unchanged third full curriculum paired with an explicit-state-only arm.

The original full coordinator, worker, core and optimizer remain unmodified.
"""
import argparse,fcntl,json,multiprocessing as mp,shutil,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
import run_first_return_curriculum as full
from policy_search_math import SearchTransform,SEARCH_CONFIG,spatial_basis
from fly_connectome.control.projection import FixedPolicyProjection
from fly_connectome.vision import VisualStateEncoder
from return_metrics import aggregate

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'docs/reconstruction/2026-10-21-forecast-comparison-protocol.md'
ANCESTOR=Path('/workspace/m1b-experiments/pong-first-return-curriculum-20261021-r1')
ANCESTOR_AUDIT=Path('/workspace/m1b-experiments/first-return-curriculum-parent-audit-20261021')
FULL_AUDIT=Path('/workspace/m1b-experiments/third-full-parent-audit-20261021')
ROOTS=dict(training_seed=1782731441,optimizer_seed=947621083,development_seed=1662073981,
           development_tag=29110,random_action_seed=719456287)
FORECAST_START=1792;FORECAST_END=3848;BOOTSTRAP_SEED=194870295
SIMPLES=tuple(n for n in full.CONTROLS if n!='initial_restricted')
EXECUTED=('endpoint','initial_restricted','phase0','blank')
sha=full.sha;atomic_json=full.atomic_json


def fit_state_only(raw,layout,dimension):
    """Unlabeled fit: mask the BASIS before centering/support/whitening."""
    if layout['future']!=(1792,3840) or layout['future_outside']!=(3840,3848):
        raise ValueError('explicit forecast layout differs')
    basis,bias=spatial_basis(layout,dimension)
    basis[:,FORECAST_START:FORECAST_END]=0.
    projected=np.asarray(raw@basis.T);present=projected[:,bias]
    if not np.array_equal(present,np.ones(len(present))):raise ValueError('present-frame calibration required')
    means=projected.mean(axis=0);means[bias]=0.
    centered=basis-means[:,None]*basis[bias][None,:]
    values=np.asarray(raw@centered.T);gram=values.T@values/len(values)
    eigenvalues,vectors=np.linalg.eigh(gram)
    tolerance=SEARCH_CONFIG['rank_relative_tolerance']*float(eigenvalues.max())
    supported=eigenvalues>tolerance;vectors=vectors[:,supported]
    whitening=(vectors/np.sqrt(eigenvalues[supported]+SEARCH_CONFIG['ridge']))@vectors.T
    matrix=whitening@centered;z=np.asarray(raw@matrix.T)
    if np.count_nonzero(matrix[:,FORECAST_START:FORECAST_END]):raise ValueError('forecast coefficients must be exactly zero')
    info=dict(config=dict(SEARCH_CONFIG),calibration_rows=len(values),basis_dimension=len(basis),
        supported_rank=int(supported.sum()),free_action_contrasts=2*int(supported.sum()),
        retained_frame_bias_basis_index=bias,raw_frame_bias_coordinate=layout['metadata'][0]+3,
        eigenvalues=eigenvalues.tolist(),rank_threshold=tolerance,
        maximum_whitening_gain=float(1/np.sqrt(eigenvalues[supported].min()+SEARCH_CONFIG['ridge'])),
        calibration_feature_rms=float(np.sqrt(np.mean(z*z))),
        calibration_energy_quantiles=np.quantile(np.sum(z*z,axis=1),[0,.5,.99,1]).tolist(),
        projected_bias_energy=float(gram[bias,bias]),
        masked_raw_columns=[FORECAST_START,FORECAST_END],mask_applied='basis before centering and whitening',
        missingness='all-zero maps to zero; unchanged frame-present centering semantics',
        policy_class='state-only linear DCT3 supported subspace; current beliefs/velocities/relations retained')
    return SearchTransform(matrix,info),z


def recipe(replication):
    lineage=None
    if replication:
        lineage=dict(directory=str(ANCESTOR),source_commit=json.loads((ANCESTOR/'launch.json').read_text())['source_commit'],
            manifest_sha256=sha(ANCESTOR/'manifest.json'),summary_sha256=sha(ANCESTOR/'summary.json'),
            completion_verification_sha256=sha(ANCESTOR/'completion-verification.json'),
            parent_audit_directory=str(ANCESTOR_AUDIT),parent_audit_summary_sha256=sha(ANCESTOR_AUDIT/'summary.json'))
    return dict(version=full.DEFAULT_RECIPE['version'],**ROOTS,
        protocol_file=str(PROTOCOL.relative_to(ROOT)),replication_of=lineage)


def joint_expected(root):
    return dict(version='one-pair-explicit-forecast-comparison-v1',source=full.prior.source_signature(),
        runtime=full.prior.runtime_signature(),model_hash=full.prior.MODEL_HASH,vision_configuration=full.prior.VISION_CONFIGURATION,
        roots=ROOTS,protocol_hash=sha(PROTOCOL),wrapper_hash=sha(__file__),
        original_coordinator_hash=sha(full.__file__),original_math_hash=sha(full.engine.MATH),
        original_worker_hash=sha(full.engine.WORKER),seed_exclusions_hash=sha(root/'seed-exclusions.json'),
        helper_hashes={name:sha(module.__file__) for name,module in (('phase',full.phase_engine),
            ('engine',full.engine),('prior',full.prior),('metrics',__import__('return_metrics')),
            ('physical',full.prior.physical),('completion',full.completion),('seed',full.seeds))},
        serve_seeds=full.serve_schedule(json.loads((root/'seed-exclusions.json').read_text())['ids'],
            development_seed=ROOTS['development_seed'],tag=ROOTS['development_tag']),
        shared_stream_exception='Both arms intentionally share this frozen pre-arm snapshot and roots; no redraw after full',
        full_fixed_files={n:sha(root/'full'/n) for n in ('search-transform.npy','transform-fit.json','radial-cap.json','projection.json')},
        state_fixed_files={n:sha(root/'state-only'/n) for n in ('search-transform.npy','transform-fit.json','radial-cap.json','projection.json')},
        state_only_executed=list(EXECUTED),reused_simple_controls=list(SIMPLES),
        forecast_bootstrap_seed=BOOTSTRAP_SEED,acquisition_bootstrap_seed=193870291,
        calibration_source_hash=sha(full.engine.CACHE),new_calibration_observations=0,
        per_arm_training_cameras=200000,per_arm_training_seconds=2400.,per_arm_total_seconds=3600.,
        final_streams_expanded=False,reserved_final_seeds=list(full.engine.FINAL))


def stage(root):
    """One joint snapshot/cohort registration before either arm starts."""
    root.mkdir(parents=True,exist_ok=True)
    full.seeds.snapshot_exclusions(root)
    excluded=json.loads((root/'seed-exclusions.json').read_text())['ids']
    if set(ROOTS.values())&set(excluded):raise ValueError('fresh joint root/tag collision')
    for name in ('full','state-only'):
        d=root/name;d.mkdir(exist_ok=True)
        for filename,value in (('seed-exclusions.json',json.loads((root/'seed-exclusions.json').read_text())),
                               ('seed-recipe.json',recipe(name=='full'))):
            path=d/filename
            if path.exists() and json.loads(path.read_text())!=value:raise ValueError('joint snapshot/recipe changed')
            atomic_json(path,value)
    full.stage(root/'full')
    raw=full.engine.read_calibration(full.engine.CACHE);encoder=VisualStateEncoder(64,64,.02)
    transform,z=fit_state_only(raw,encoder.layout,encoder.dimension)
    state=root/'state-only'
    full.immutable_array(state/'search-transform.npy',transform.matrix)
    fit=state/'transform-fit.json'
    if fit.exists() and json.loads(fit.read_text())!=transform.diagnostics:raise ValueError('fixed state-only fit changed')
    atomic_json(fit,transform.diagnostics)
    cap=state/'radial-cap.json'
    if cap.exists() and sha(cap)!=sha(root/'full'/'radial-cap.json'):raise ValueError('state-only cap changed')
    if not cap.exists():shutil.copyfile(root/'full'/'radial-cap.json',cap)
    descriptor=json.loads((root/'full'/'projection.json').read_text())
    descriptor.update(matrix_file=str((state/'search-transform.npy').resolve()),file_sha256=sha(state/'search-transform.npy'),
                      array_sha256=full.array_hash(transform.matrix))
    descriptor['calibration'].update(fit_file=str(fit.resolve()),fit_sha256=sha(fit))
    path=state/'projection.json'
    if path.exists() and json.loads(path.read_text())!=descriptor:raise ValueError('state-only projection changed')
    atomic_json(path,descriptor);FixedPolicyProjection(4389,descriptor).validate_embedded()
    expected=joint_expected(root);path=root/'comparison-registration.json'
    if path.exists() and json.loads(path.read_text())!=expected:raise ValueError('joint comparison changed')
    atomic_json(path,expected)


def validate_joint(root):
    actual=json.loads((root/'comparison-registration.json').read_text())
    if actual!=joint_expected(root):raise ValueError('joint source/inputs/roster/roots changed')
    for name in ('full','state-only'):
        if sha(root/name/'seed-exclusions.json')!=actual['seed_exclusions_hash']:
            raise ValueError('joint exclusion snapshot changed')
        if json.loads((root/name/'seed-recipe.json').read_text())!=recipe(name=='full'):
            raise ValueError('joint arm recipe changed')
    return actual


def prepare_state(root):
    joint=validate_joint(root);d=root/'state-only';descriptor=json.loads((d/'projection.json').read_text())
    projection=FixedPolicyProjection(4389,descriptor);projection.validate_embedded()
    matrix=np.load(d/'search-transform.npy',allow_pickle=False)
    if matrix.shape!=(214,4389) or np.count_nonzero(matrix[:,FORECAST_START:FORECAST_END]):
        raise ValueError('exact state-only forecast zeroing required')
    expected=dict(source=full.prior.source_signature(),runtime=full.prior.runtime_signature(),model_hash=full.prior.MODEL_HASH,
        vision_configuration=full.prior.VISION_CONFIGURATION,driver_hash=sha(full.engine.WORKER),coordinator_hash=sha(__file__),
        original_coordinator_hash=sha(full.__file__),joint_registration_hash=sha(root/'comparison-registration.json'),
        seed_recipe=recipe(False),seed_recipe_hash=sha(d/'seed-recipe.json'),protocol_hash=sha(PROTOCOL),
        **ROOTS,serve_seeds=joint['serve_seeds'],phases=list(full.PHASES),search_config=SEARCH_CONFIG,
        fixed_files={n:sha(d/n) for n in ('search-transform.npy','transform-fit.json','radial-cap.json','projection.json','seed-exclusions.json')},
        training_wall_seconds=full.TRAIN_SECONDS,total_wall_seconds=full.TOTAL_SECONDS,
        evaluation_wall_seconds=full.EVAL_SECONDS,evaluation_camera_cap=128000,
        reused_control_jobs=320,reused_control_maximum_cameras=160000,
        primary_controls=list(full.CONTROLS),secondary_only=['phase0','blank'],
        calibration_source_hash=sha(full.engine.CACHE),reused_calibration_rows=1024,new_calibration_observations=0,
        workers=4,gc_worker_mode=full.prior.GC_WORKER_MODE,long_bank_entries=0,context_fallback_steps=4,
        reserved_final_seeds=list(full.engine.FINAL),executed_policies=list(EXECUTED),
        reused_control_directory=str(root/'full'),reused_policies=list(SIMPLES))
    path=d/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('state-only allocation changed')
    else:
        manifest=dict(**expected,started_wall_ns=time.time_ns(),jobs=[]);atomic_json(path,manifest)
    transform=SearchTransform(matrix,json.loads((d/'transform-fit.json').read_text()))
    z=np.asarray(full.engine.read_calibration(full.engine.CACHE)@matrix.T)
    rng=np.random.default_rng(ROOTS['optimizer_seed']);theta=SEARCH_CONFIG['initial_scale']*full.prior.sample_directions(rng,z,1)[0]
    full.immutable_array(d/'theta_initial.npy',theta);full.weights_file(d,'policy_initial.npy',transform.mapped_weights(theta))
    return manifest,transform,z,theta,rng.bit_generator.state


def reuse_controls(root):
    """Verify ORIGINAL full evidence in place; never copy mismatched pins."""
    d=root/'full';manifest=json.loads((d/'manifest.json').read_text())
    completed=json.loads((d/'summary.json').read_text())
    if completed['status']!='complete':raise ValueError('full completion required before state-only clock; acquisition pass is not required')
    own=json.loads((d/'completion-verification.json').read_text())
    parent=json.loads((FULL_AUDIT/'summary.json').read_text())
    if not own['passed'] or not parent['passed']:raise ValueError('verify full raw evidence before state-only')
    evidence_hashes=own.get('evaluation_evidence_hashes',{})
    if (set(evidence_hashes)!={'manifest.json','summary.json','return-annotations.json'}
            or evidence_hashes!=parent.get('evidence_file_hashes')
            or any(sha(d/n)!=h for n,h in evidence_hashes.items())):
        raise ValueError('audited shared evaluation evidence changed')
    if manifest['source']!=full.prior.source_signature() or manifest['serve_seeds']!=validate_joint(root)['serve_seeds']:
        raise ValueError('shared control source/cohort differs')
    declared=full.evaluation_jobs(manifest);annotations=json.loads((d/'return-annotations.json').read_text())
    result={};proof=[]
    for policy in SIMPLES:
        group=[j for j in declared if j['policy']==policy];rallies=[]
        for job in group:
            report=full.completion.complete_report(d,job)
            if report is None:continue
            actual=next(j for j in manifest['jobs'] if j['id']==job['id'])
            if actual!=job:raise ValueError('shared control job/config/action identity differs')
            annotation=annotations[job['id']]
            if len(annotation['rallies'])!=1:raise ValueError('one complete shared control rally required')
            rallies.extend(annotation['rallies'])
            proof.append(dict(policy=policy,job=job['id'],directory=str(d),report_hash=sha(d/(job['id']+'.json')),
                completion_hash=sha(d/(job['id']+'-completion.json')),camera_rows=report['transitions'],
                annotations_hash=sha(d/'return-annotations.json'),full_summary_hash=sha(d/'summary.json'),
                own_verification_hash=sha(d/'completion-verification.json'),parent_audit_hash=sha(FULL_AUDIT/'summary.json')))
        result[policy]=rallies
    return result,proof


def comparison(positive,negative,draws=10000):
    if len(positive)!=64 or len(negative)!=64:return dict(complete=False)
    for a,b in zip(positive,negative):
        one,two=a['first_incoming'],b['first_incoming']
        if (None if one is None else {k:v for k,v in one.items() if k!='success'})!=(None if two is None else {k:v for k,v in two.items() if k!='success'}):
            raise ValueError('paired forecast comparison geometry differs')
    index=np.random.default_rng(BOOTSTRAP_SEED).integers(64,size=(draws,64));rates=[];points=[]
    for data in (positive,negative):
        hits=np.array([r['own_returns'] for r in data]);opp=np.array([r['own_opportunities'] for r in data])
        first=np.array([bool(r['first_incoming'] and r['first_incoming']['success']) for r in data]);present=np.array([r['first_incoming'] is not None for r in data])
        with np.errstate(divide='ignore',invalid='ignore'):
            rates.append((first[index].sum(1)/present[index].sum(1),hits[index].sum(1)/opp[index].sum(1)))
            points.append((float(first.sum()/present.sum()),float(hits.sum()/opp.sum())))
    first=rates[0][0]-rates[1][0];all_returns=rates[0][1]-rates[1][1]
    valid=np.isfinite(first)&np.isfinite(all_returns)
    ci=np.quantile(first[valid],[.025,.975]).tolist() if valid.any() else None
    ca=np.quantile(all_returns[valid],[.025,.975]).tolist() if valid.any() else None
    return dict(complete=True,first_incoming_lift=points[0][0]-points[1][0],all_opportunity_lift=points[0][1]-points[1][1],
        first_incoming_lift_ci95=ci,all_opportunity_lift_ci95=ca,invalid_bootstrap_draws=int((~valid).sum()),
        bootstrap_seed=BOOTSTRAP_SEED,bootstrap_draws=draws,serve_clusters=64,
        positive_development_signal=bool(valid.all() and ci[0]>0 and ca[0]>0),
        reverse_development_signal=bool(valid.all() and ci[1]<0 and ca[1]<0),
        scope='One training pair, development only; not across-training-seed inference or equivalence')


def supervise_state(root):
    d=root/'state-only';torch.set_num_threads(1)
    with (d/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controls,proof=reuse_controls(root)
        atomic_json(d/'reused-control-proof.json',proof)
        manifest,transform,z,theta,rng=prepare_state(root);start=manifest['started_wall_ns'];anchor=start;index=0;phases=[]
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=full.prior.initialize_worker) as pool:
            for spec in full.PHASES:
                phase,state,current=full.initialize_phase(d,spec,theta,rng,index,anchor)
                frozen,theta=full.run_phase(phase,spec,transform,z,state,current,index,pool,
                    min(anchor/1e9+spec['wall_seconds'],start/1e9+full.TRAIN_SECONDS))
                phases.append(dict(phase=spec['name'],**frozen))
                if not frozen['valid']:break
                if spec['first_hit_cutoff']:full.weights_file(d,'policy_phase0.npy',transform.mapped_weights(theta))
                rng,index,anchor=frozen['next_rng_state'],frozen['next_global_generation'],frozen['frozen_wall_ns']
            full.weights_file(d,'policy_endpoint.npy',transform.mapped_weights(theta))
            jobs=[j for j in full.evaluation_jobs(manifest) if j['policy'] in EXECUTED] if len(phases)==2 and all(p['valid'] for p in phases) else []
            full.engine.register_jobs(d,jobs)
            evaluated=full.engine.execute_jobs(d,jobs,pool,deadline=min(start/1e9+full.TOTAL_SECONDS,anchor/1e9+full.EVAL_SECONDS)-180.,
                camera_cap=128000,initial_consumed=full.engine.counts(d,json.loads((d/'manifest.json').read_text()))['fresh_evaluation'],**full.OPTIONS)
        annotations=full.prior.physical.annotate(d,jobs);atomic_json(d/'return-annotations.json',annotations)
        by_policy=dict(controls)
        for policy in EXECUTED:
            by_policy[policy]=[r for j in jobs if j['policy']==policy and j['id'] in annotations for r in annotations[j['id']]['rallies']]
        groups={name:dict(complete=len(rallies)==64,**aggregate(rallies)) for name,rallies in by_policy.items()}
        primary,secondary=full.evidence(by_policy,include_owninitial=True)
        live_full=json.loads((root/'full'/'return-annotations.json').read_text())
        full_jobs=json.loads((root/'full'/'manifest.json').read_text())['jobs']
        contrasts={}
        for policy in ('endpoint','initial_restricted','phase0'):
            positive=[r for j in full_jobs if j.get('policy')==policy and j['id'] in live_full for r in live_full[j['id']]['rallies']]
            contrasts[policy]=comparison(positive,by_policy[policy])
        counts=dict(training_fitness=0,fresh_evaluation=0)
        for path in (d,*[d/p['name'] for p in full.PHASES]):
            if (path/'manifest.json').exists():
                for k,v in full.engine.counts(path,json.loads((path/'manifest.json').read_text())).items():counts[k]+=v
        assert counts['training_fitness']<=200000 and counts['fresh_evaluation']<=128000
        elapsed=time.time()-start/1e9;eval_elapsed=time.time()-anchor/1e9
        overall=elapsed<=full.TOTAL_SECONDS;eval_met=eval_elapsed<=full.EVAL_SECONDS
        shared_complete=all(len(controls[p])==64 for p in SIMPLES)
        all_cohorts_complete=len(groups)==9 and all(g['complete'] for g in groups.values())
        complete=bool(len(phases)==2 and all(p['valid'] for p in phases) and evaluated['complete'] and overall and eval_met and shared_complete and all_cohorts_complete)
        full_summary=json.loads((root/'full'/'summary.json').read_text())
        eligible=bool(complete and full_summary['status']=='complete' and full_summary['overall_budget_met']
            and full_summary['evaluation_budget_met'] and contrasts['endpoint'].get('complete',False))
        summary=dict(status='complete' if complete else 'partial',
            forecast_evidence_eligible=eligible,
            within_allocation_forecast_development_signal=bool(eligible and contrasts['endpoint'].get('positive_development_signal',False)),
            within_allocation_reverse_forecast_signal=bool(eligible and contrasts['endpoint'].get('reverse_development_signal',False)),
            overall_budget_met=overall,evaluation_budget_met=eval_met,shared_controls_complete=shared_complete,evaluation_started_wall_ns=anchor,evaluation_wall_seconds=eval_elapsed,
            phases=phases,counts=counts,evaluation=evaluated,groups=groups,paired_evidence=primary,secondary=secondary,
            full_minus_state_only=contrasts,reused_control_cameras=sum(p['camera_rows'] for p in proof),
            reused_control_proof_hash=sha(d/'reused-control-proof.json'),wall_seconds=elapsed,
            selected_checkpoint='last full phase1 update; no best/checkpoint selection',
            final_holdout_unused=list(full.engine.FINAL),scope='One state-only training arm, paired with unchanged third full arm')
        atomic_json(d/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--stage',action='store_true');parser.add_argument('--arm',choices=('full','state-only'))
    args=parser.parse_args()
    if args.stage:stage(args.directory)
    else:
        validate_joint(args.directory)
        if args.arm=='full':full.supervise(args.directory/'full')
        elif args.arm=='state-only':supervise_state(args.directory)
        else:parser.error('--arm is required for execution')
