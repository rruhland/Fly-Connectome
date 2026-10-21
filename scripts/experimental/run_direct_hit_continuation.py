"""Fixed same-class continuation with independently durable progress diagnostics."""
import argparse,fcntl,json,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
import run_direct_hit_policy_search as prior
import run_frozen_hit_confirmation as confirmation
from policy_search_math import generation_spec,mirrored_update
from return_metrics import aggregate,paired_evidence

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'docs/reconstruction/2026-10-21-direct-hit-continuation-protocol.md'
LINEAGE=confirmation.LINEAGE
TRAINING_SEED=1563241907;DEV_SEED=1718230093;ACTION_SEED=1880921331
TRAIN_CAMERAS=160000;TRAIN_SECONDS=2100.;EVAL_SECONDS=1200.;TOTAL_SECONDS=3300.
EVAL_CAMERAS=288000;MAX_UPDATES=16;SERVES=64
CONTROLS=prior.CONTROLS
POLICIES=('endpoint',*CONTROLS,'theta9','blank')
sha=prior.sha;atomic_json=prior.atomic_json;array_hash=prior.array_hash
weights_file=prior.weights_file;immutable_array=prior.immutable_array
completion=prior.completion;engine=prior.engine


def counts(directory,manifest):
    result=dict(training_fitness=0,training_anchor=0,fresh_evaluation=0)
    for job in manifest['jobs']:result[job['purpose']]+=engine.sample_count(directory,job['id'])
    return result


def training_count(directory):
    c=counts(directory,json.loads((directory/'manifest.json').read_text()))
    return c['training_fitness']+c['training_anchor']


def stage(directory):
    confirmation.stage(directory)
    endpoint=json.loads((LINEAGE/'training-endpoint.json').read_text())
    pending=json.loads((LINEAGE/'g09-definition.json').read_text())
    if (endpoint['completed_generations']!=9 or endpoint['next_global_generation']!=10
            or not endpoint['unused_draw_skipped'] or endpoint['unused_cameras']!=14312
            or endpoint['next_rng_state']!=pending['rng_state_after_draw']):
        raise ValueError('inherited skipped-draw boundary differs')
    theta=np.load(LINEAGE/'theta_at9.npy',allow_pickle=False)
    if array_hash(theta)!=endpoint['theta_hash']:raise ValueError('inherited theta differs')
    immutable_array(directory/'theta_at0.npy',theta)
    weights_file(directory,'policy_at0.npy',theta@np.load(directory/'search-transform.npy',allow_pickle=False))
    if sha(directory/'policy_at0.npy')!=sha(directory/'endpoint.npy'):
        raise ValueError('local initial policy must be inherited theta9')
    proof=dict(endpoint=endpoint,endpoint_hash=sha(LINEAGE/'training-endpoint.json'),
        skipped_definition_hash=sha(LINEAGE/'g09-definition.json'),
        original_learned_updates=9,first_global_generation=10,unused_rows_reused=0)
    path=directory/'boundary-proof.json'
    if path.exists() and json.loads(path.read_text())!=proof:raise ValueError('continuation boundary changed')
    atomic_json(path,proof)


def validate_initial_boundary(directory):
    proof=json.loads((directory/'boundary-proof.json').read_text())
    endpoint=json.loads((LINEAGE/'training-endpoint.json').read_text())
    pending=json.loads((LINEAGE/'g09-definition.json').read_text())
    if (proof!=dict(endpoint=endpoint,endpoint_hash=sha(LINEAGE/'training-endpoint.json'),
            skipped_definition_hash=sha(LINEAGE/'g09-definition.json'),
            original_learned_updates=9,first_global_generation=10,unused_rows_reused=0)
            or endpoint['completed_generations']!=9 or endpoint['next_global_generation']!=10
            or not endpoint['unused_draw_skipped'] or endpoint['unused_cameras']!=14312
            or endpoint['next_rng_state']!=pending['rng_state_after_draw']):
        raise ValueError('inherited skipped-draw boundary differs')
    theta=np.load(directory/'theta_at0.npy',allow_pickle=False)
    if (array_hash(theta)!=endpoint['theta_hash']
            or not np.array_equal(theta,np.load(LINEAGE/'theta_at9.npy',allow_pickle=False))
            or not np.array_equal(theta@np.load(directory/'search-transform.npy',allow_pickle=False),
                                  np.load(directory/'endpoint.npy',allow_pickle=False))
            or sha(directory/'policy_at0.npy')!=sha(directory/'endpoint.npy')):
        raise ValueError('inherited theta9 mapping differs')
    return endpoint


def serve_schedule(excluded):
    rng=np.random.default_rng(np.random.SeedSequence([DEV_SEED,29106]));result=[]
    excluded=set(excluded)|set(engine.FINAL)
    while len(result)<SERVES:
        value=int(rng.integers(1,2147483647))
        if value not in excluded and value not in result:result.append(value)
    return result


def prepare(directory):
    fixed=('lineage-proof.json','boundary-proof.json','seed-exclusions.json','endpoint.npy','endpoint.json',
        'initial-restricted.npy','initial-restricted.json','theta_at0.npy','policy_at0.npy','policy_at0.json',
        'search-transform.npy','transform-fit.json','radial-cap.json','projection.json')
    if any(not (directory/name).exists() for name in fixed):raise ValueError('stage complete frozen inputs before clock')
    boundary=validate_initial_boundary(directory)
    proof=json.loads((directory/'lineage-proof.json').read_text())
    if (proof['source']!=prior.source_signature() or proof['runtime']!=prior.runtime_signature()
            or proof['model_hash']!=prior.MODEL_HASH or proof['vision_configuration']!=prior.VISION_CONFIGURATION
            or any(sha(directory/name)!=digest for name,digest in proof['mapped_files'].items())):
        raise ValueError('frozen lineage source/runtime/files differ')
    excluded=json.loads((directory/'seed-exclusions.json').read_text())['ids']
    expected=dict(source=prior.source_signature(),runtime=prior.runtime_signature(),model_hash=prior.MODEL_HASH,
        vision_configuration=prior.VISION_CONFIGURATION,coordinator_hash=sha(__file__),protocol_hash=sha(PROTOCOL),
        driver_hash=sha(engine.WORKER),engine_hash=sha(engine.__file__),prior_helper_hash=sha(prior.__file__),
        confirmation_helper_hash=sha(confirmation.__file__),math_hash=sha(engine.MATH),
        completion_helper_hash=sha(completion.__file__),physical_helper_hash=sha(prior.physical.__file__),
        metrics_hash=sha(Path(prior.physical.__file__).with_name('return_metrics.py')),
        fixed_files={name:sha(directory/name) for name in fixed},serve_seeds=serve_schedule(excluded),
        training_seed=TRAINING_SEED,development_seed=DEV_SEED,random_action_seed=ACTION_SEED,
        training_camera_cap=TRAIN_CAMERAS,evaluation_camera_cap=EVAL_CAMERAS,maximum_new_updates=MAX_UPDATES,
        training_wall_seconds=TRAIN_SECONDS,evaluation_wall_seconds=EVAL_SECONDS,total_wall_seconds=TOTAL_SECONDS,
        primary_controls=list(CONTROLS),secondary_only=['theta9','blank'],workers=4,
        gc_worker_mode=prior.GC_WORKER_MODE,long_bank_entries=0,context_fallback_steps=4,
        reserved_final_seeds=list(engine.FINAL),anchor_cadence='even global generations; pre-update center and theta9')
    path=directory/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if any(manifest[k]!=v for k,v in expected.items()):raise ValueError('registered continuation changed')
    else:
        manifest=dict(**expected,started_wall_ns=time.time_ns(),jobs=[]);atomic_json(path,manifest)
    matrix=np.load(directory/'search-transform.npy',allow_pickle=False)
    z=np.asarray(engine.read_calibration(engine.CACHE)@matrix.T)
    state_path=directory/'search-state.json'
    if not state_path.exists():
        atomic_json(state_path,dict(completed_updates=0,next_global_generation=boundary['next_global_generation'],
            theta_file='theta_at0.npy',theta_hash=boundary['theta_hash'],rng_state=boundary['next_rng_state'],
            pending_diagnostics=None))
    state=json.loads(state_path.read_text());theta=np.load(directory/state['theta_file'],allow_pickle=False)
    if array_hash(theta)!=state['theta_hash']:raise ValueError('optimizer checkpoint differs')
    committed_fitness_rows(directory,state)
    weights_file(directory,f"policy_at{state['completed_updates']}.npy",theta@matrix)
    return manifest,z,state,theta


def prepare_population(directory,z,state,theta,excluded):
    g=state['next_global_generation'];n=state['completed_updates'];label=f'g{g:02d}'
    directions,candidates,serves,after=generation_spec(theta,state['rng_state'],g,z,
        training_seed=TRAINING_SEED,serves_per_generation=6,excluded_seeds=excluded)
    matrix=np.load(directory/'search-transform.npy',allow_pickle=False)
    immutable_array(directory/(label+'-directions.npy'),directions);jobs=[]
    for d in range(8):
        for sign in range(2):
            prefix=f'{label}_d{d}_{"plus" if sign==0 else "minus"}';filename=prefix+'.npy'
            weights_file(directory,filename,candidates[d,sign]@matrix)
            for s,seed in enumerate(serves):
                job=prior.rollout_job(f'{prefix}_s{s}',seed,filename,'training_fitness')
                job.update(direction_index=d,sign_index=sign,serve_index=s);jobs.append(job)
    diagnostics=[];groups={};readouts={}
    if g%2==0:
        center=f'{label}_center_before_update{9+n}.npy';weights_file(directory,center,theta@matrix)
        references={'theta9':'endpoint.npy','current_center':center}
        if n==0 and sha(directory/center)!=sha(directory/'endpoint.npy'):
            raise ValueError('initial diagnostic identity differs')
        for role,filename in references.items():
            if n==0 and role=='current_center':groups[role]=groups['theta9'];continue
            names=[]
            for s,seed in enumerate(serves):
                job=prior.rollout_job(f'{label}_anchor_{role}_s{s}',seed,filename,'training_anchor')
                job.update(anchor_role=role,serve_index=s);diagnostics.append(job);names.append(job['id'])
            groups[role]=names
        readouts=dict(source_local_update=n,source_cumulative_update=9+n,
            source_theta_file=state['theta_file'],source_theta_hash=state['theta_hash'],
            reference_cumulative_update=9,reference_weight_hash=sha(directory/'endpoint.npy'),
            center_weight_file=center,center_weight_hash=sha(directory/center),initial_identity_reused=n==0)
    definition=dict(generation=n,global_generation=g,input_theta_hash=array_hash(theta),
        input_rng_state=state['rng_state'],rng_state_after_draw=after,directions_hash=array_hash(directions),
        training_serve_seeds=serves,jobs=jobs,diagnostic_jobs=diagnostics,anchor_groups=groups,anchor_readouts=readouts)
    path=directory/(label+'-definition.json')
    if path.exists() and json.loads(path.read_text())!=definition:raise ValueError('pinned population differs')
    atomic_json(path,definition);engine.register_jobs(directory,[*jobs,*diagnostics])
    return definition,directions


def finish_fitness(directory,definition,directions,theta,z):
    if array_hash(theta)!=definition['input_theta_hash']:raise ValueError('fitness input theta differs')
    returns,hits,reports=prior.population_fitness(directory,definition)
    proposed,optimizer=mirrored_update(theta,directions,returns,z)
    n=definition['generation']+1;g=definition['global_generation'];label=f'g{g:02d}'
    immutable_array(directory/f'theta_at{n}.npy',proposed)
    weights_file(directory,f'policy_at{n}.npy',proposed@np.load(directory/'search-transform.npy',allow_pickle=False))
    result=dict(generation=n-1,global_generation=g,returns=returns.tolist(),
        hit_counts_by_direction_sign_serve=hits.astype(int).tolist(),optimizer=optimizer,rollouts=reports,
        new_theta_hash=array_hash(proposed),cumulative_learned_updates=9+n,fitness='mean own hits only; anchors excluded')
    path=directory/(label+'-result.json')
    if path.exists() and json.loads(path.read_text())!=result:raise ValueError('completed fitness result differs')
    atomic_json(path,result)
    state=dict(completed_updates=n,next_global_generation=g+1,theta_file=f'theta_at{n}.npy',
        theta_hash=array_hash(proposed),rng_state=definition['rng_state_after_draw'],
        pending_diagnostics=label if definition['diagnostic_jobs'] else None)
    atomic_json(directory/'search-state.json',state)
    return state,proposed


def finish_diagnostics(directory,definition):
    values={}
    for role,names in definition['anchor_groups'].items():
        hits=[]
        for name in names:
            job=next(j for j in definition['diagnostic_jobs'] if j['id']==name)
            report=completion.complete_report(directory,job)
            if report is None:raise ValueError('partial diagnostic cannot be complete')
            hits.append(report['hits'])
        values[role]=hits
    differences=np.asarray(values['current_center'])-np.asarray(values['theta9'])
    result=dict(global_generation=definition['global_generation'],source=definition['anchor_readouts'],
        hits=values,paired_hit_difference=differences.tolist(),mean_hit_difference=float(differences.mean()),
        diagnostic_only=True,used_for_fitness_or_selection=False)
    path=directory/f"g{definition['global_generation']:02d}-anchor-result.json"
    if path.exists() and json.loads(path.read_text())!=result:raise ValueError('completed anchor differs')
    atomic_json(path,result)


def logged_failures(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    purposes={j['id']:j['purpose'] for j in manifest['jobs']}
    path=directory/'rollout-costs.jsonl'
    failed={purposes[c['job']] for c in map(json.loads,path.read_text().splitlines()) if not c['ok']} if path.exists() else set()
    return 'training_fitness' in failed,'training_anchor' in failed


def committed_fitness_rows(directory,state):
    """Only optimizer-state-committed results count; orphan results never do."""
    n=state['completed_updates'];initial=json.loads((directory/'boundary-proof.json').read_text())['endpoint']
    previous_hash=initial['theta_hash'];rng=initial['next_rng_state'];used=0
    if type(n) is not int or not 0<=n<=MAX_UPDATES:raise ValueError('invalid committed update count')
    for local in range(n):
        g=10+local;label=f'g{g:02d}'
        definition=json.loads((directory/(label+'-definition.json')).read_text())
        result=json.loads((directory/(label+'-result.json')).read_text())
        theta=np.load(directory/f'theta_at{local+1}.npy',allow_pickle=False)
        if (definition['generation']!=local or definition['global_generation']!=g
                or definition['input_theta_hash']!=previous_hash or definition['input_rng_state']!=rng
                or result['generation']!=local or result['global_generation']!=g
                or result['new_theta_hash']!=array_hash(theta)
                or [r['job'] for r in result['rollouts']]!=[j['id'] for j in definition['jobs']]):
            raise ValueError('committed fitness lineage differs')
        if any(type(r['transitions']) is not int or r['transitions']<0 for r in result['rollouts']):
            raise ValueError('invalid committed fitness counters')
        used+=sum(r['transitions'] for r in result['rollouts'])
        previous_hash=result['new_theta_hash'];rng=definition['rng_state_after_draw']
        if definition['diagnostic_jobs'] and local<n-1 and not (directory/(label+'-anchor-result.json')).exists():
            raise ValueError('advanced past unfinished diagnostics')
    last=f'g{9+n:02d}' if n else None
    pending=state['pending_diagnostics']
    if (state['next_global_generation']!=10+n or state['theta_file']!=f'theta_at{n}.npy'
            or state['theta_hash']!=previous_hash or state['rng_state']!=rng
            or pending not in (None,last) or (not n and pending is not None)):
        raise ValueError('optimizer boundary differs from committed fitness')
    if n:
        definition=json.loads((directory/(last+'-definition.json')).read_text())
        if (pending and not definition['diagnostic_jobs']) or (
                definition['diagnostic_jobs'] and pending is None and not (directory/(last+'-anchor-result.json')).exists()):
            raise ValueError('optimizer diagnostic phase differs')
    return used


def freeze_fields(directory,state):
    used=committed_fitness_rows(directory,state);g=state['next_global_generation']
    path=directory/f'g{g:02d}-definition.json';draw=json.loads(path.read_text()) if path.exists() else None
    if draw and (draw['generation']!=state['completed_updates'] or draw['global_generation']!=g
            or draw['input_rng_state']!=state['rng_state'] or draw['input_theta_hash']!=state['theta_hash']):
        raise ValueError('pending transition draw differs')
    if (directory/f'g{g:02d}-result.json').exists():raise ValueError('unreconciled fitness result at frozen boundary')
    current=counts(directory,json.loads((directory/'manifest.json').read_text()))
    if not 0<=used<=current['training_fitness'] or current['training_fitness']+current['training_anchor']>TRAIN_CAMERAS:
        raise ValueError('training camera counters differ')
    fitness_failed,anchor_failed=logged_failures(directory)
    return dict(new_updates=state['completed_updates'],cumulative_learned_updates=9+state['completed_updates'],
        theta_file=state['theta_file'],theta_hash=state['theta_hash'],fitness_cameras=current['training_fitness'],
        used_fitness_cameras=used,unused_partial_fitness_cameras=current['training_fitness']-used,
        diagnostic_cameras=current['training_anchor'],
        total_training_phase_cameras=current['training_fitness']+current['training_anchor'],
        pending_diagnostics=state['pending_diagnostics'],anchor_failure=anchor_failed,fitness_failure=fitness_failed,
        diagnostic_status='failed' if anchor_failed else 'partial' if state['pending_diagnostics'] else 'complete',
        unused_draw_skipped=bool(draw),next_global_generation=g+int(bool(draw)),
        next_rng_state=draw['rng_state_after_draw'] if draw else state['rng_state'],
        valid=state['completed_updates']>=2 and not fitness_failed)


def execute(directory,jobs,pool,deadline,camera_cap,current):
    return engine.execute_jobs(directory,jobs,pool,deadline=deadline,camera_cap=camera_cap,
        initial_consumed=current,completion_report=completion.complete_report,completion_pin=completion.pin_completion)


def train(directory,pool,manifest,z,state,theta):
    marker=directory/'training-endpoint.json'
    if marker.exists():
        frozen=json.loads(marker.read_text())
        if any(frozen[k]!=v for k,v in freeze_fields(directory,state).items()):
            raise ValueError('frozen endpoint differs from optimizer')
        return frozen,state,theta
    deadline=manifest['started_wall_ns']/1e9+TRAIN_SECONDS
    excluded=(*json.loads((directory/'seed-exclusions.json').read_text())['ids'],*manifest['serve_seeds'])
    fitness_failed,anchor_failed=logged_failures(directory)
    reason='prior logged fitness failure' if fitness_failed else None
    while not fitness_failed:
        if state['pending_diagnostics']:
            label=state['pending_diagnostics'];definition=json.loads((directory/(label+'-definition.json')).read_text())
            if anchor_failed:reason='prior logged diagnostic failure';break
            result=execute(directory,definition['diagnostic_jobs'],pool,deadline,TRAIN_CAMERAS,training_count(directory))
            if not result['complete']:
                anchor_failed=result['failed'];reason='diagnostic failure' if anchor_failed else 'training camera/wall budget during diagnostics';break
            finish_diagnostics(directory,definition)
            state=dict(state,pending_diagnostics=None);atomic_json(directory/'search-state.json',state)
        if state['completed_updates']>=MAX_UPDATES:reason='maximum complete new updates';break
        # Reconcile a complete pinned population before any remaining-budget test.
        # A result-to-state crash consumes no new observations and never drops or
        # double-applies its update, even when the original budget is exhausted.
        pending=directory/f"g{state['next_global_generation']:02d}-definition.json"
        if pending.exists():
            definition,directions=prepare_population(directory,z,state,theta,excluded)
            if all(completion.complete_report(directory,j) is not None for j in definition['jobs']):
                state,theta=finish_fitness(directory,definition,directions,theta,z);continue
        if TRAIN_CAMERAS-training_count(directory)<200 or deadline-time.time()<=25:
            reason='training camera/wall budget';break
        definition,directions=prepare_population(directory,z,state,theta,excluded)
        result=execute(directory,definition['jobs'],pool,deadline,TRAIN_CAMERAS,training_count(directory))
        if not result['complete']:
            fitness_failed=result['failed'];reason='fitness failure' if fitness_failed else 'training camera/wall budget during fitness'
            atomic_json(directory/'incomplete-fitness-generation.json',dict(global_generation=definition['global_generation'],**result));break
        state,theta=finish_fitness(directory,definition,directions,theta,z)
        print(json.dumps(dict(new_updates=state['completed_updates'],global_generation=definition['global_generation'],
            training_phase_cameras=training_count(directory))),flush=True)
    frozen=dict(**freeze_fields(directory,state),training_stop=reason,frozen_wall_ns=time.time_ns())
    atomic_json(marker,frozen);return frozen,state,theta


def evaluation_jobs(directory,manifest,state):
    jobs=[];new=f"policy_at{state['completed_updates']}.npy"
    for policy in POLICIES:
        for i,seed in enumerate(manifest['serve_seeds']):
            trained=policy in ('endpoint','initial_restricted','theta9','blank')
            filename=('initial-restricted.npy' if policy=='initial_restricted' else 'endpoint.npy' if policy=='theta9' else new) if trained else None
            kind='trained' if trained else 'random' if policy.startswith('random') else policy
            action_seed=int(np.random.SeedSequence([ACTION_SEED,DEV_SEED,i]).generate_state(1)[0]) if kind=='random' else None
            if action_seed in engine.FINAL:raise ValueError('reserved action seed collision')
            job=prior.rollout_job(f'eval_{policy}_s{i:02d}',seed,filename,'fresh_evaluation',kind=kind,
                action_seed=action_seed,hold=10 if policy=='random10' else 1)
            job['config']['blank_camera']=policy=='blank';job.update(policy=policy,serve_index=i);jobs.append(job)
    return jobs


def evidence(by_policy):
    primary=paired_evidence(by_policy,arms=('endpoint',),expected_serves=SERVES,controls=CONTROLS)
    secondary={}
    for name in ('theta9','blank'):
        value=paired_evidence(by_policy,arms=('endpoint','theta9','blank'),expected_serves=SERVES,
            controls=CONTROLS,comparison=('endpoint',name)).get('paired_arm_comparison',dict(complete=False))
        secondary[name]=value
    v=secondary['theta9']
    extra=bool(v.get('invalid_bootstrap_draws')==0 and v.get('first_incoming_lift_ci95') and v['first_incoming_lift_ci95'][0]>0
        and v['all_opportunity_lift_ci95'][0]>0)
    return primary,secondary,extra


def supervise(directory):
    torch.set_num_threads(1)
    with (directory/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest,z,state,theta=prepare(directory);start=manifest['started_wall_ns']/1e9
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=prior.initialize_worker) as pool:
            frozen,state,theta=train(directory,pool,manifest,z,state,theta)
            jobs=evaluation_jobs(directory,manifest,state) if frozen['valid'] else [];engine.register_jobs(directory,jobs)
            phase=directory/'evaluation-phase.json'
            if not phase.exists():atomic_json(phase,dict(started_wall_ns=time.time_ns()))
            eval_start=json.loads(phase.read_text())['started_wall_ns']/1e9
            deadline=min(start+TOTAL_SECONDS,eval_start+EVAL_SECONDS)-180.
            evaluated=execute(directory,jobs,pool,deadline,EVAL_CAMERAS,
                counts(directory,json.loads((directory/'manifest.json').read_text()))['fresh_evaluation'])
        annotations=prior.physical.annotate(directory,jobs);atomic_json(directory/'return-annotations.json',annotations)
        by_policy={};groups={}
        for policy in POLICIES:
            rallies=[r for j in jobs if j['policy']==policy and j['id'] in annotations for r in annotations[j['id']]['rallies']]
            by_policy[policy]=rallies;groups[policy]=dict(complete=len(rallies)==SERVES,**aggregate(rallies))
        primary,secondary,extra=evidence(by_policy);actual=counts(directory,json.loads((directory/'manifest.json').read_text()))
        elapsed=time.time()-start;eval_elapsed=time.time()-eval_start;within=elapsed<=TOTAL_SECONDS and eval_elapsed<=EVAL_SECONDS
        assert actual['training_fitness']+actual['training_anchor']<=TRAIN_CAMERAS and actual['fresh_evaluation']<=EVAL_CAMERAS
        summary=dict(status='complete' if frozen['valid'] and evaluated['complete'] and within else 'partial',
            counts=actual,training_endpoint=frozen,evaluation=evaluated,groups=groups,paired_evidence=primary,
            secondary_comparisons=secondary,extra_continuation_learning_evidence=extra,overall_budget_met=within,
            wall_seconds=elapsed,evaluation_wall_seconds=eval_elapsed,selected_checkpoint=f"policy_at{state['completed_updates']}.npy",
            selection='last complete learning update; anchors never select',final_holdout_unused=list(engine.FINAL),
            scope='One trained lineage; no independent replication, score mastery or completed M1B')
        atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--stage',action='store_true');args=parser.parse_args()
    if args.stage:stage(args.directory)
    else:supervise(args.directory)
