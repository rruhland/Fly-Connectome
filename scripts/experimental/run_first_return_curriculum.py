"""Fixed first-return to full-rally, reward-only mirrored policy curriculum."""
import argparse,fcntl,json,multiprocessing as mp,shutil,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
import run_direct_hit_policy_search as prior
import run_frozen_hit_confirmation as seeds
from policy_search_math import SearchTransform,generation_spec,mirrored_update,function_rms
from fly_connectome.control.projection import FixedPolicyProjection
from return_metrics import aggregate,paired_evidence

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'docs/reconstruction/2026-10-21-first-return-curriculum-protocol.md'
TRAINING_SEED=1518615419;OPTIMIZER_SEED=1097410397;DEV_SEED=1746610937;ACTION_SEED=1297478891
SERVES=64;TRAIN_SECONDS=2400.;TOTAL_SECONDS=3600.;EVAL_SECONDS=1200.;EVAL_CAMERAS=288000
PHASES=(dict(name='phase0-first-return',opponent_speed=.65,generations=16,camera_cap=120000,
             wall_seconds=1440.,serves=12,first_hit_cutoff=True),
        dict(name='phase1-full-rally',opponent_speed=.65,generations=16,camera_cap=80000,
             wall_seconds=960.,serves=6,first_hit_cutoff=False))
CONTROLS=prior.CONTROLS;POLICIES=('endpoint',*CONTROLS,'phase0','blank')
engine=prior.engine;completion=prior.completion;phase_engine=prior.phase_engine
sha=prior.sha;atomic_json=prior.atomic_json;array_hash=prior.array_hash
immutable_array=prior.immutable_array;weights_file=prior.weights_file
OPTIONS=dict(completion_report=completion.complete_report,completion_pin=completion.pin_completion)


DEFAULT_RECIPE=dict(version='first-return-seed-recipe-v1',training_seed=TRAINING_SEED,
    optimizer_seed=OPTIMIZER_SEED,development_seed=DEV_SEED,random_action_seed=ACTION_SEED,
    development_tag=29107,protocol_file=str(PROTOCOL.relative_to(ROOT)),replication_of=None)


def registration(directory):
    """Seed/registration metadata only; no algorithm or feature knobs."""
    path=directory/'seed-recipe.json'
    value=json.loads(path.read_text()) if path.exists() else dict(DEFAULT_RECIPE)
    if set(value)!=set(DEFAULT_RECIPE) or value['version']!=DEFAULT_RECIPE['version']:
        raise ValueError('invalid immutable seed recipe')
    names=('training_seed','optimizer_seed','development_seed','random_action_seed')
    if (any(type(value[n]) is not int or not 0<value[n]<2147483647 for n in (*names,'development_tag'))
            or len({value[n] for n in names})!=4 or {value[n] for n in names}&set(engine.FINAL)
            or not isinstance(value['protocol_file'],str)):
        raise ValueError('invalid fresh recipe seeds/tag')
    protocol=(ROOT/value['protocol_file']).resolve()
    if not protocol.is_relative_to(ROOT/'docs/reconstruction') or not protocol.is_file():
        raise ValueError('recipe requires an archived repository protocol')
    lineage=value['replication_of']
    if lineage is not None:
        fields={'directory','source_commit','manifest_sha256','summary_sha256',
                'completion_verification_sha256','parent_audit_directory','parent_audit_summary_sha256'}
        if not isinstance(lineage,dict) or set(lineage)!=fields:raise ValueError('invalid replication provenance')
        ancestor=Path(lineage['directory']);audit=Path(lineage['parent_audit_directory'])/'summary.json'
        paths={'manifest_sha256':ancestor/'manifest.json','summary_sha256':ancestor/'summary.json',
            'completion_verification_sha256':ancestor/'completion-verification.json','parent_audit_summary_sha256':audit}
        if any(sha(file)!=lineage[key] for key,file in paths.items()):raise ValueError('replication lineage bytes changed')
        summary=json.loads((ancestor/'summary.json').read_text());manifest=json.loads((ancestor/'manifest.json').read_text())
        if (summary['status']!='complete' or not summary['paired_evidence']['arms']['endpoint']['promising_development_signal']
                or json.loads((ancestor/'launch.json').read_text())['source_commit']!=lineage['source_commit']
                or not json.loads(audit.read_text())['passed']
                or not json.loads((ancestor/'completion-verification.json').read_text())['passed']
                or manifest['source']!=prior.source_signature() or manifest['runtime']!=prior.runtime_signature()
                or manifest['model_hash']!=prior.MODEL_HASH or manifest['vision_configuration']!=prior.VISION_CONFIGURATION
                or not set(value[n] for n in names).isdisjoint(manifest[n] for n in names)):
            raise ValueError('verified independent same-condition replication required')
    return value,protocol


def serve_schedule(excluded,*,development_seed=DEV_SEED,tag=29107):
    rng=np.random.default_rng(np.random.SeedSequence([development_seed,tag]));result=[]
    excluded=set(excluded)|set(engine.FINAL)
    while len(result)<SERVES:
        seed=int(rng.integers(1,2147483647))
        if seed not in excluded and seed not in result:result.append(seed)
    return result


def stage(directory):
    """Immutable score-free inputs and metadata only; no manifest clock."""
    directory.mkdir(parents=True,exist_ok=True)
    recipe,_=registration(directory)
    for name in ('search-transform.npy','transform-fit.json','radial-cap.json'):
        target=directory/name;source=engine.LINEAGE/name
        if target.exists() and sha(target)!=sha(source):raise ValueError('fixed transform/cap differs')
        if not target.exists():shutil.copyfile(source,target)
    descriptor=json.loads((seeds.LINEAGE/'projection.json').read_text())
    descriptor['matrix_file']=str((directory/'search-transform.npy').resolve())
    path=directory/'projection.json'
    if path.exists() and json.loads(path.read_text())!=descriptor:raise ValueError('projection descriptor differs')
    atomic_json(path,descriptor);FixedPolicyProjection(4389,descriptor).validate_embedded()
    seeds.snapshot_exclusions(directory)
    excluded=json.loads((directory/'seed-exclusions.json').read_text())['ids']
    if {recipe[n] for n in ('training_seed','optimizer_seed','development_seed','random_action_seed')} & set(excluded):
        raise ValueError('fresh declared seed-root collision')


def prepare(directory):
    fixed=('search-transform.npy','transform-fit.json','radial-cap.json','projection.json','seed-exclusions.json')
    if any(not (directory/n).exists() for n in fixed):raise ValueError('stage inputs before launch clock')
    for name in fixed[:3]:
        if sha(directory/name)!=sha(engine.LINEAGE/name):raise ValueError('fixed transform/cap changed')
    FixedPolicyProjection(4389,json.loads((directory/'projection.json').read_text())).validate_embedded()
    excluded=json.loads((directory/'seed-exclusions.json').read_text())['ids']
    recipe,protocol=registration(directory)
    expected=dict(source=prior.source_signature(),runtime=prior.runtime_signature(),model_hash=prior.MODEL_HASH,
        vision_configuration=prior.VISION_CONFIGURATION,driver_hash=sha(engine.WORKER),
        coordinator_hash=sha(__file__),engine_hash=sha(engine.__file__),phase_helper_hash=sha(phase_engine.__file__),
        prior_helper_hash=sha(prior.__file__),seed_helper_hash=sha(seeds.__file__),math_hash=sha(engine.MATH),
        completion_helper_hash=sha(completion.__file__),physical_helper_hash=sha(prior.physical.__file__),
        metrics_hash=sha(Path(prior.physical.__file__).with_name('return_metrics.py')),protocol_hash=sha(protocol),
        registration_protocol=recipe['protocol_file'],default_protocol_hash=sha(PROTOCOL),
        seed_recipe=recipe,seed_recipe_hash=sha(directory/'seed-recipe.json') if (directory/'seed-recipe.json').exists() else None,
        fixed_files={n:sha(directory/n) for n in fixed},search_config=prior.SEARCH_CONFIG,phases=list(PHASES),
        training_seed=recipe['training_seed'],optimizer_seed=recipe['optimizer_seed'],
        development_seed=recipe['development_seed'],development_tag=recipe['development_tag'],random_action_seed=recipe['random_action_seed'],
        serve_seeds=serve_schedule(excluded,development_seed=recipe['development_seed'],tag=recipe['development_tag']),training_wall_seconds=TRAIN_SECONDS,total_wall_seconds=TOTAL_SECONDS,
        evaluation_wall_seconds=EVAL_SECONDS,evaluation_camera_cap=EVAL_CAMERAS,primary_controls=list(CONTROLS),
        secondary_only=['phase0','blank'],workers=4,gc_worker_mode=prior.GC_WORKER_MODE,
        calibration_source_hash=sha(engine.CACHE),reused_calibration_rows=1024,new_calibration_observations=0,
        long_bank_entries=0,context_fallback_steps=4,reserved_final_seeds=list(engine.FINAL))
    path=directory/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('registered curriculum changed')
    else:
        manifest=dict(**expected,started_wall_ns=time.time_ns(),jobs=[]);atomic_json(path,manifest)
    transform=SearchTransform(np.load(directory/'search-transform.npy'),json.loads((directory/'transform-fit.json').read_text()))
    z=np.asarray(engine.read_calibration(engine.CACHE)@transform.matrix.T)
    rng=np.random.default_rng(recipe['optimizer_seed']);theta=prior.SEARCH_CONFIG['initial_scale']*prior.sample_directions(rng,z,1)[0]
    immutable_array(directory/'theta_initial.npy',theta);weights_file(directory,'policy_initial.npy',transform.mapped_weights(theta))
    return manifest,transform,z,theta,rng.bit_generator.state


def initialize_phase(root,spec,theta,rng,index,start):
    directory=root/spec['name'];directory.mkdir(exist_ok=True)
    parent=json.loads((root/'manifest.json').read_text())
    expected=dict(source=parent['source'],runtime=parent['runtime'],driver_hash=parent['driver_hash'],
        root_manifest_started_wall_ns=parent['started_wall_ns'],root_coordinator_hash=parent['coordinator_hash'],
        phase=spec,phase_started_wall_ns=start,global_generation_offset=index,training_seed=parent['training_seed'],
        input_theta_hash=array_hash(theta),input_rng_state=rng,
        excluded_training_seeds=sorted(set(parent['serve_seeds'])|set(json.loads((root/'seed-exclusions.json').read_text())['ids'])))
    path=directory/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if any(manifest[k]!=v for k,v in expected.items()):raise ValueError('phase lineage/clock/config differs')
    else:atomic_json(path,dict(**expected,jobs=[]))
    for name in ('search-transform.npy','radial-cap.json','projection.json'):
        if not (directory/name).exists():shutil.copyfile(root/name,directory/name)
        if sha(directory/name)!=sha(root/name):raise ValueError('phase fixed artifact differs')
    immutable_array(directory/'theta_at0.npy',theta);weights_file(directory,'policy_at0.npy',theta@np.load(root/'search-transform.npy'))
    state_path=directory/'search-state.json'
    initial=dict(completed_generations=0,theta_file='theta_at0.npy',theta_hash=array_hash(theta),
        rng_state=rng,initial_policy_hash=sha(directory/'policy_at0.npy'))
    if not state_path.exists():atomic_json(state_path,initial)
    state=json.loads(state_path.read_text());current=np.load(directory/state['theta_file'])
    committed_rows(directory,state)
    if array_hash(current)!=state['theta_hash']:raise ValueError('phase theta checkpoint differs')
    return directory,state,current


def prepare_population(directory,transform,z,state,theta,global_offset,speed):
    manifest=json.loads((directory/'manifest.json').read_text());spec=manifest['phase']
    if global_offset!=manifest['global_generation_offset'] or speed!=.65:raise ValueError('phase generation boundary differs')
    local=state['completed_generations'];g=global_offset+local;label=f'g{local:02d}'
    directions,candidates,serves,after=generation_spec(theta,state['rng_state'],g,z,training_seed=manifest['training_seed'],
        serves_per_generation=spec['serves'],excluded_seeds=manifest['excluded_training_seeds'])
    immutable_array(directory/(label+'-directions.npy'),directions);jobs=[]
    for d in range(8):
        for sign in range(2):
            prefix=f'{label}_d{d}_{"plus" if sign==0 else "minus"}';filename=prefix+'.npy'
            weights_file(directory,filename,transform.mapped_weights(candidates[d,sign]))
            for s,seed in enumerate(serves):
                job=prior.rollout_job(f'{prefix}_s{s}',seed,filename,'training_fitness')
                job.update(direction_index=d,sign_index=sign,serve_index=s)
                if spec['first_hit_cutoff']:job['first_hit_cutoff']=True
                jobs.append(job)
    definition=dict(generation=local,global_generation=g,input_theta_hash=array_hash(theta),
        input_rng_state=state['rng_state'],rng_state_after_draw=after,directions_hash=array_hash(directions),
        first_hit_cutoff=spec['first_hit_cutoff'],training_serve_seeds=serves,jobs=jobs)
    path=directory/(label+'-definition.json')
    if path.exists() and json.loads(path.read_text())!=definition:raise ValueError('pending population differs')
    atomic_json(path,definition);engine.register_jobs(directory,jobs)
    return definition,directions


def finish_population(directory,definition,directions,theta,z):
    if definition['first_hit_cutoff']:
        for job in definition['jobs']:
            r=completion.complete_report(directory,job)
            if (not r or r.get('task_definition')!='first-own-hit-task-v1' or r['hits'] not in (0,1)
                    or r['episodes']!=r['wins']+r['losses']+r['truncations']+r['task_stops']
                    or r['task_stops']!=r['hits']):raise ValueError('invalid first-return task fitness')
    if definition['input_theta_hash']!=array_hash(theta):raise ValueError('population input theta differs')
    returns,hits,reports=prior.population_fitness(directory,definition)
    proposed,diagnostics=mirrored_update(theta,directions,returns,z)
    local=definition['generation'];count=local+1
    immutable_array(directory/f'theta_at{count}.npy',proposed)
    weights_file(directory,f'policy_at{count}.npy',proposed@np.load(directory/'search-transform.npy'))
    contrasts=hits[:,0,:]-hits[:,1,:];norm=function_rms(proposed,z)
    result=dict(generation=local,global_generation=definition['global_generation'],returns=returns.tolist(),
        hit_counts_by_direction_sign_serve=hits.astype(int).tolist(),optimizer=diagnostics,rollouts=reports,
        new_theta_hash=array_hash(proposed),first_hit_cutoff=definition['first_hit_cutoff'],
        total_hits=int(hits.sum()),hit_bearing_rollouts=int(np.count_nonzero(hits)),
        per_serve_total_hits=hits.sum(axis=(0,1)).astype(int).tolist(),
        per_serve_nonzero_pairs=np.count_nonzero(contrasts,axis=0).tolist(),
        nonzero_paired_mean_contrasts=int(np.count_nonzero(returns[:,0]-returns[:,1])),
        policy_function_rms=norm,noise_policy_rms_ratio=prior.SEARCH_CONFIG['noise_scale']/norm if norm else None,
        fitness='mean binary first own hit' if definition['first_hit_cutoff'] else 'mean total own hits')
    path=directory/f'g{local:02d}-result.json'
    if path.exists() and json.loads(path.read_text())!=result:raise ValueError('completed population differs')
    atomic_json(path,result)
    state=dict(completed_generations=count,theta_file=f'theta_at{count}.npy',theta_hash=array_hash(proposed),
        rng_state=definition['rng_state_after_draw'],initial_policy_hash=sha(directory/'policy_at0.npy'))
    atomic_json(directory/'search-state.json',state)
    return state,proposed,result


def committed_rows(directory,state):
    manifest=json.loads((directory/'manifest.json').read_text());spec=manifest['phase']
    n=state['completed_generations'];previous=manifest['input_theta_hash'];rng=manifest['input_rng_state'];used=0
    if type(n) is not int or not 0<=n<=spec['generations']:raise ValueError('invalid phase update count')
    for local in range(n):
        definition=json.loads((directory/f'g{local:02d}-definition.json').read_text())
        result=json.loads((directory/f'g{local:02d}-result.json').read_text())
        theta=np.load(directory/f'theta_at{local+1}.npy')
        if (definition['generation']!=local or definition['global_generation']!=manifest['global_generation_offset']+local
                or definition['input_theta_hash']!=previous or definition['input_rng_state']!=rng
                or definition['first_hit_cutoff']!=spec['first_hit_cutoff']
                or len(definition['jobs'])!=16*spec['serves'] or result['generation']!=local
                or result['new_theta_hash']!=array_hash(theta)
                or [r['job'] for r in result['rollouts']]!=[j['id'] for j in definition['jobs']]):
            raise ValueError('committed phase lineage differs')
        if any(type(r['transitions']) is not int or r['transitions']<0 for r in result['rollouts']):
            raise ValueError('invalid committed camera counts')
        used+=sum(r['transitions'] for r in result['rollouts']);previous=result['new_theta_hash'];rng=definition['rng_state_after_draw']
    if (state['theta_file']!=f'theta_at{n}.npy' or state['theta_hash']!=previous or state['rng_state']!=rng
            or state['initial_policy_hash']!=sha(directory/'policy_at0.npy')):raise ValueError('optimizer boundary differs')
    return used


def failed_tasks(directory):
    path=directory/'rollout-costs.jsonl'
    return any(not c['ok'] for c in map(json.loads,path.read_text().splitlines())) if path.exists() else False


def validate_frozen(directory,state,frozen):
    manifest=json.loads((directory/'manifest.json').read_text());used=committed_rows(directory,state)
    observed=engine.counts(directory,manifest)['training_fitness']
    rng,index,skipped=phase_engine.next_boundary(directory,state,manifest['global_generation_offset'])
    expected=dict(completed_generations=state['completed_generations'],theta_file=state['theta_file'],
        theta_hash=state['theta_hash'],next_rng_state=rng,next_global_generation=index,unused_draw_skipped=skipped,
        observed_cameras=observed,used_cameras=used,unused_cameras=observed-used,
        failed=failed_tasks(directory),valid=not failed_tasks(directory) and state['completed_generations']>=2)
    if not 0<=used<=observed<=manifest['phase']['camera_cap'] or any(frozen[k]!=v for k,v in expected.items()):
        raise ValueError('frozen phase counters/lineage differ')
    if (directory/f"g{state['completed_generations']:02d}-result.json").exists():
        raise ValueError('unreconciled population at frozen boundary')
    return frozen


def run_phase(directory,spec,transform,z,state,theta,index,pool,deadline):
    marker=directory/'training-endpoint.json'
    if marker.exists():return validate_frozen(directory,state,json.loads(marker.read_text())),theta
    # A complete, pinned population is reconciled without new work BEFORE budget
    # checks, including a crash after result write but before optimizer commit.
    local=state['completed_generations'];path=directory/f'g{local:02d}-definition.json'
    if path.exists() and not failed_tasks(directory):
        definition,directions=prepare_population(directory,transform,z,state,theta,index,.65)
        if all(completion.complete_report(directory,j) is not None for j in definition['jobs']):
            state,theta,_=finish_population(directory,definition,directions,theta,z)
    frozen,theta=phase_engine.run_phase(directory,spec,transform,z,state,theta,index,pool,deadline,
        population_builder=prepare_population,population_updater=finish_population,execution_options=OPTIONS)
    return validate_frozen(directory,json.loads((directory/'search-state.json').read_text()),frozen),theta


def evaluation_jobs(manifest):
    jobs=[]
    for policy in POLICIES:
        trained=policy in ('endpoint','initial_restricted','phase0','blank')
        filename=('policy_initial.npy' if policy=='initial_restricted' else 'policy_phase0.npy' if policy=='phase0' else 'policy_endpoint.npy') if trained else None
        for i,seed in enumerate(manifest['serve_seeds']):
            kind='trained' if trained else ('random' if policy.startswith('random') else policy)
            action_seed=int(np.random.SeedSequence([manifest['random_action_seed'],manifest['development_seed'],i]).generate_state(1)[0]) if kind=='random' else None
            if action_seed in engine.FINAL:raise ValueError('reserved action seed collision')
            job=prior.rollout_job(f'eval_{policy}_s{i:02d}',seed,filename,'fresh_evaluation',kind=kind,
                action_seed=action_seed,hold=10 if policy=='random10' else 1)
            job['config']['blank_camera']=policy=='blank';job.update(policy=policy,serve_index=i);jobs.append(job)
    return jobs


def evidence(by_policy,*,include_owninitial=False):
    primary=paired_evidence(by_policy,arms=('endpoint',),expected_serves=SERVES,controls=CONTROLS)
    comparisons={}
    for name,pair in (('phase0_minus_initial',('phase0','initial_restricted')),
                      ('endpoint_minus_phase0',('endpoint','phase0')),('live_minus_blank',('endpoint','blank'))):
        comparisons[name]=paired_evidence(by_policy,arms=pair,expected_serves=SERVES,controls=CONTROLS,
            comparison=pair).get('paired_arm_comparison',dict(complete=False))
    if include_owninitial:
        comparisons['endpoint_minus_initial']=paired_evidence(by_policy,arms=('endpoint','initial_restricted'),
            expected_serves=SERVES,controls=CONTROLS,comparison=('endpoint','initial_restricted')).get(
                'paired_arm_comparison',dict(complete=False))
    return primary,comparisons


def supervise(directory):
    torch.set_num_threads(1)
    with (directory/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest,transform,z,theta,rng=prepare(directory);start=manifest['started_wall_ns'];index=0;anchor=start;phases=[]
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=prior.initialize_worker) as pool:
            for spec in PHASES:
                phase,state,current=initialize_phase(directory,spec,theta,rng,index,anchor)
                frozen,theta=run_phase(phase,spec,transform,z,state,current,index,pool,
                    min(anchor/1e9+spec['wall_seconds'],start/1e9+TRAIN_SECONDS))
                phases.append(dict(phase=spec['name'],**frozen))
                if not frozen['valid']:break
                if spec['first_hit_cutoff']:weights_file(directory,'policy_phase0.npy',transform.mapped_weights(theta))
                rng,index,anchor=frozen['next_rng_state'],frozen['next_global_generation'],frozen['frozen_wall_ns']
            weights_file(directory,'policy_endpoint.npy',transform.mapped_weights(theta))
            jobs=evaluation_jobs(manifest) if len(phases)==2 and all(p['valid'] for p in phases) else []
            engine.register_jobs(directory,jobs)
            evaluated=engine.execute_jobs(directory,jobs,pool,
                deadline=min(start/1e9+TOTAL_SECONDS,anchor/1e9+EVAL_SECONDS)-180.,camera_cap=EVAL_CAMERAS,
                initial_consumed=engine.counts(directory,json.loads((directory/'manifest.json').read_text()))['fresh_evaluation'],**OPTIONS)
        annotations=prior.physical.annotate(directory,jobs);atomic_json(directory/'return-annotations.json',annotations)
        by_policy={};groups={}
        for policy in POLICIES:
            rallies=[r for j in jobs if j['policy']==policy and j['id'] in annotations for r in annotations[j['id']]['rallies']]
            by_policy[policy]=rallies;groups[policy]=dict(complete=len(rallies)==SERVES,**aggregate(rallies))
        primary,secondary=evidence(by_policy,include_owninitial=manifest['seed_recipe']['replication_of'] is not None)
        counts=dict(training_fitness=0,fresh_evaluation=0)
        for path in [directory,*[directory/p['name'] for p in PHASES]]:
            if (path/'manifest.json').exists():
                for k,v in engine.counts(path,json.loads((path/'manifest.json').read_text())).items():counts[k]+=v
        assert counts['training_fitness']<=200000 and counts['fresh_evaluation']<=EVAL_CAMERAS
        elapsed=time.time()-start/1e9;evaluation_elapsed=time.time()-anchor/1e9
        evaluation_budget_met=evaluation_elapsed<=EVAL_SECONDS
        complete=bool(jobs and evaluated['complete'] and elapsed<=TOTAL_SECONDS and evaluation_budget_met)
        summary=dict(status='complete' if complete else 'partial',overall_budget_met=elapsed<=TOTAL_SECONDS,
            evaluation_started_wall_ns=anchor,evaluation_wall_seconds=evaluation_elapsed,evaluation_budget_met=evaluation_budget_met,
            phases=phases,counts=counts,evaluation=evaluated,groups=groups,paired_evidence=primary,secondary=secondary,
            selected_checkpoint='last full phase1 update; phase0 fixed boundary secondary only; never best',
            wall_seconds=elapsed,final_holdout_unused=list(engine.FINAL),
            scope=('One independent whole-procedure seed replication; retain every outcome; not score mastery or complete M1B'
                if manifest['seed_recipe']['replication_of'] is not None else
                'One-seed first-return/full-rally reward curriculum; not independent replication, score mastery or complete M1B'))
        atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--stage',action='store_true');args=parser.parse_args()
    if args.stage:stage(args.directory)
    else:supervise(args.directory)
