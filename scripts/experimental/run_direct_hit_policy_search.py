"""Direct hit-return search composed with the existing durable phase engine."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import fcntl,gc,json,multiprocessing as mp,time
from pathlib import Path
import numpy as np
import torch
import run_dct_sarsa_pong as completion
import run_pong_policy_search_continuation as engine
import run_stationary_curriculum as phase_engine
import run_reliable_return_learning as physical
from policy_search_math import SEARCH_CONFIG,SearchTransform,generation_spec,sample_directions,mirrored_update,function_rms
from fly_connectome.control.session import source_signature,MODEL_HASH,VISION_CONFIGURATION
from fly_connectome.control.durable import runtime_signature
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.projection import FixedPolicyProjection
from return_metrics import aggregate,paired_evidence

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'docs/reconstruction/2026-10-21-direct-hit-policy-search-protocol.md'
TRAINING_SEED=1563241907;OPTIMIZER_SEED=1337923911
TRAIN_SECONDS=2100.;TOTAL_SECONDS=2700.;TRAIN_CAMERAS=160000;SERVES=32
SPEC=dict(name='direct-hit',opponent_speed=.65,generations=16,camera_cap=TRAIN_CAMERAS,wall_seconds=TRAIN_SECONDS)
CONTROLS=('initial_restricted','initial','up','noop','random1','random10')
GC_WORKER_MODE='freeze-imports-once-before-first-session:gc-enabled-v1'
sha=engine.sha;array_hash=engine.array_hash;atomic_json=engine.atomic_json
immutable_array=engine.immutable_array;weights_file=engine.weights_file


def initialize_worker():
    """Freeze imported objects once; subsequent candidate sessions stay collectible."""
    if not gc.isenabled():raise RuntimeError('registered search requires enabled cyclic collection')
    torch.set_num_threads(1)
    gc.collect()
    gc.freeze()


def serve_schedule():
    values=np.random.default_rng(np.random.SeedSequence([933715231,29104])).integers(1,2147483647,size=SERVES)
    result=[int(x) for x in values]
    if len(set(result))!=SERVES or set(result)&set(engine.FINAL):raise ValueError('development stream collision')
    return result


def pins(directory):
    return dict(source=source_signature(),runtime=runtime_signature(),model_hash=MODEL_HASH,
        vision_configuration=VISION_CONFIGURATION,driver_hash=sha(engine.WORKER),
        coordinator_hash=sha(__file__),engine_hash=sha(engine.__file__),phase_helper_hash=sha(phase_engine.__file__),
        completion_helper_hash=sha(completion.__file__),physical_helper_hash=sha(physical.__file__),
        math_hash=sha(engine.MATH),metrics_hash=sha(Path(physical.__file__).with_name('return_metrics.py')),
        protocol_hash=sha(PROTOCOL),search_config=SEARCH_CONFIG,training_seed=TRAINING_SEED,
        optimizer_seed=OPTIMIZER_SEED,serve_seeds=serve_schedule(),spec=SPEC,total_wall_seconds=TOTAL_SECONDS,
        training_workers=4,evaluation_workers=4,gc_worker_mode=GC_WORKER_MODE,
        evaluation_camera_cap=112000,serves_per_generation=6,
        transform_hash=sha(directory/'search-transform.npy'),fit_hash=sha(directory/'transform-fit.json'),
        cap_hash=sha(directory/'radial-cap.json'),projection_hash=sha(directory/'projection.json'),
        calibration_source_hash=sha(engine.CACHE),reused_calibration_rows=1024,new_calibration_observations=0,
        long_bank_entries=0,context_fallback_steps=4,reserved_final_seeds=list(engine.FINAL))


def prepare(directory):
    directory.mkdir(parents=True,exist_ok=True)
    for name in ('search-transform.npy','transform-fit.json','radial-cap.json'):
        if sha(directory/name)!=sha(engine.LINEAGE/name):raise ValueError('exact fixed transform/fit/cap required')
    descriptor=json.loads((directory/'projection.json').read_text())
    projection=FixedPolicyProjection(4389,descriptor);projection.validate_embedded()
    if (descriptor['file_sha256']!=sha(directory/'search-transform.npy')
            or descriptor['shape']!=[214,4389] or descriptor['max_norm2']!=256.
            or descriptor['calibration']['fit_sha256']!=sha(directory/'transform-fit.json')):
        raise ValueError('mapped-policy projection provenance differs')
    expected=pins(directory);path=directory/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if any(manifest[k]!=v for k,v in expected.items()):raise ValueError('registered hit search changed')
    else:
        manifest=dict(**expected,started_wall_ns=time.time_ns(),jobs=[],purpose='direct hit-only mirrored policy search')
        atomic_json(path,manifest)
    transform=SearchTransform(np.load(directory/'search-transform.npy',allow_pickle=False),
        json.loads((directory/'transform-fit.json').read_text()))
    calibration=np.asarray(engine.read_calibration(engine.CACHE)@transform.matrix.T)
    rng=np.random.default_rng(OPTIMIZER_SEED);theta=SEARCH_CONFIG['initial_scale']*sample_directions(rng,calibration,1)[0]
    immutable_array(directory/'theta_at0.npy',theta);weights_file(directory,'policy_at0.npy',transform.mapped_weights(theta))
    state_path=directory/'search-state.json'
    if not state_path.exists():atomic_json(state_path,dict(completed_generations=0,theta_file='theta_at0.npy',
        theta_hash=array_hash(theta),rng_state=rng.bit_generator.state,initial_policy_hash=sha(directory/'policy_at0.npy')))
    state=json.loads(state_path.read_text());theta=np.load(directory/state['theta_file'],allow_pickle=False)
    if array_hash(theta)!=state['theta_hash'] or state['initial_policy_hash']!=sha(directory/'policy_at0.npy'):
        raise ValueError('optimizer checkpoint differs')
    weights_file(directory,f"policy_at{state['completed_generations']}.npy",transform.mapped_weights(theta))
    return manifest,transform,calibration,state,theta


def rollout_job(name,seed,weights,purpose,*,kind='trained',action_seed=None,hold=1):
    config=asdict(PongEvaluationConfig(seed,episodes=1,max_episode_ticks=1200,policy_kind=kind,
        action_seed=action_seed,skip_baseline_perception=kind!='trained',opponent_speed=.65,action_hold_frames=hold))
    return dict(id=name,kind='evaluate',purpose=purpose,config=config,weights_from=weights,
        scaling_from='radial-cap.json' if kind=='trained' else None,compact_reports=True)


def prepare_population(directory,transform,z,state,theta,global_offset,speed):
    if speed!=.65 or global_offset!=0:raise ValueError('fixed from-scratch task required')
    prior_dev=np.random.default_rng(np.random.SeedSequence([1287749219,29103])).integers(1,2147483647,size=32)
    g=state['completed_generations'];directions,candidates,serves,after=generation_spec(theta,state['rng_state'],g,z,
        training_seed=TRAINING_SEED,serves_per_generation=6,excluded_seeds=(*serve_schedule(),*map(int,prior_dev)))
    label=f'g{g:02d}';jobs=[];immutable_array(directory/(label+'-directions.npy'),directions)
    for d in range(8):
        for sign in range(2):
            prefix=f'{label}_d{d}_{"plus" if sign==0 else "minus"}';filename=prefix+'.npy'
            weights_file(directory,filename,transform.mapped_weights(candidates[d,sign]))
            for s,seed in enumerate(serves):
                job=rollout_job(f'{prefix}_s{s}',seed,filename,'training_fitness')
                job.update(direction_index=d,sign_index=sign,serve_index=s);jobs.append(job)
    definition=dict(generation=g,global_generation=g,input_theta_hash=array_hash(theta),input_rng_state=state['rng_state'],
        directions_hash=array_hash(directions),training_serve_seeds=serves,rng_state_after_draw=after,jobs=jobs)
    path=directory/(label+'-definition.json')
    if path.exists() and json.loads(path.read_text())!=definition:raise ValueError('pending population differs')
    atomic_json(path,definition);engine.register_jobs(directory,jobs)
    return definition,directions


def population_fitness(directory,definition):
    count=len(definition['training_serve_seeds']);hits=np.zeros((8,2,count));seen=set();reports=[]
    for job in definition['jobs']:
        key=(job['direction_index'],job['sign_index'],job['serve_index'])
        if key in seen or not (0<=key[0]<8 and 0<=key[1]<2 and 0<=key[2]<count):
            raise ValueError('duplicate or invalid population cell')
        seen.add(key);r=completion.complete_report(directory,job)
        if not r or not r['complete'] or r['episodes']!=1:raise ValueError('partial population cannot update')
        value=r['hits']
        if type(value) is not int or value<0:raise ValueError('integer own-hit fitness required')
        hits[key]=value
        reports.append(dict(job=job['id'],hits=value,terminal_score=r['total_reward'],
            transitions=r['transitions'],physics_seconds=r['physics_seconds']))
    if len(seen)!=16*count:raise ValueError('incomplete population cells')
    return hits.mean(axis=2),hits,reports


def finish_population(directory,definition,directions,theta,z):
    if definition['input_theta_hash']!=array_hash(theta):raise ValueError('population input policy differs')
    returns,hits,reports=population_fitness(directory,definition)
    proposed,diagnostics=mirrored_update(theta,directions,returns,z,top=4,step_size=SEARCH_CONFIG['step_size'])
    g=definition['generation'];count=g+1
    immutable_array(directory/f'theta_at{count}.npy',proposed)
    weights_file(directory,f'policy_at{count}.npy',proposed@np.load(directory/'search-transform.npy',allow_pickle=False))
    contrasts=hits[:,0,:]-hits[:,1,:];norm=function_rms(proposed,z)
    result=dict(generation=g,returns=returns.tolist(),hit_counts_by_direction_sign_serve=hits.astype(int).tolist(),
        optimizer=diagnostics,rollouts=reports,new_theta_hash=array_hash(proposed),
        per_serve_nonzero_pairs=np.count_nonzero(contrasts,axis=0).tolist(),per_serve_total_hits=hits.sum(axis=(0,1)).astype(int).tolist(),
        nonzero_paired_mean_contrasts=int(np.count_nonzero(returns[:,0]-returns[:,1])),
        hit_bearing_rollouts=int(np.count_nonzero(hits)),total_hits=int(hits.sum()),
        policy_function_rms=norm,noise_policy_rms_ratio=SEARCH_CONFIG['noise_scale']/norm if norm else None,
        fitness='mean total own hits across six shared fresh serves; score ignored; no opportunity or teacher labels')
    path=directory/f'g{g:02d}-result.json'
    if path.exists() and json.loads(path.read_text())!=result:raise ValueError('completed population differs')
    atomic_json(path,result)
    state=dict(completed_generations=count,theta_file=f'theta_at{count}.npy',theta_hash=array_hash(proposed),
        rng_state=definition['rng_state_after_draw'],initial_policy_hash=sha(directory/'policy_at0.npy'))
    atomic_json(directory/'search-state.json',state)
    return state,proposed,result


def evaluation_jobs(state):
    jobs=[]
    for policy in ('endpoint',*CONTROLS):
        trained=policy in ('endpoint','initial_restricted')
        weights=f"policy_at{state['completed_generations']}.npy" if policy=='endpoint' else ('policy_at0.npy' if trained else None)
        for i,seed in enumerate(serve_schedule()):
            kind='trained' if trained else ('random' if policy.startswith('random') else policy)
            action_seed=int(np.random.SeedSequence([447201113,933715231,i]).generate_state(1)[0]) if kind=='random' else None
            job=rollout_job(f'eval_{policy}_s{i:02d}',seed,weights,'fresh_evaluation',kind=kind,
                action_seed=action_seed,hold=10 if policy=='random10' else 1)
            job.update(policy=policy,serve_index=i);jobs.append(job)
    return jobs


def supervise(directory):
    torch.set_num_threads(1);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest,transform,z,state,theta=prepare(directory);start=manifest['started_wall_ns']/1e9
        options=dict(completion_report=completion.complete_report,completion_pin=completion.pin_completion)
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize_worker) as pool:
            frozen,theta=phase_engine.run_phase(directory,SPEC,transform,z,state,theta,0,pool,start+TRAIN_SECONDS,
                population_builder=prepare_population,population_updater=finish_population,execution_options=options)
            state=json.loads((directory/'search-state.json').read_text());jobs=evaluation_jobs(state) if frozen['valid'] else []
            engine.register_jobs(directory,jobs)
            evaluated=engine.execute_jobs(directory,jobs,pool,deadline=start+TOTAL_SECONDS-120.,camera_cap=112000,
                initial_consumed=engine.counts(directory,json.loads((directory/'manifest.json').read_text()))['fresh_evaluation'],**options)
        actual_manifest=json.loads((directory/'manifest.json').read_text());counts=engine.counts(directory,actual_manifest)
        # Training fitness/diagnostics already use pinned raw hit reports.
        # Opportunity annotations are evaluation-only; the independent parent
        # audits training physics separately and accounts that replay cost.
        annotations=physical.annotate(directory,jobs);atomic_json(directory/'return-annotations.json',annotations)
        by_policy={};groups={}
        for policy in ('endpoint',*CONTROLS):
            rallies=[r for j in jobs if j['policy']==policy and j['id'] in annotations for r in annotations[j['id']]['rallies']]
            by_policy[policy]=rallies;groups[policy]=dict(complete=len(rallies)==SERVES,**aggregate(rallies))
        assert counts['training_fitness']<=TRAIN_CAMERAS and counts['fresh_evaluation']<=112000
        evidence=paired_evidence(by_policy,arms=('endpoint',),expected_serves=SERVES,controls=CONTROLS)
        summary=dict(status='complete' if frozen['valid'] and evaluated['complete'] else 'partial',counts=counts,
            training_endpoint=frozen,evaluation=evaluated,groups=groups,paired_evidence=evidence,
            selected_checkpoint=f"policy_at{state['completed_generations']}.npy",selection='last full generation, never best',
            wall_seconds=time.time()-start,final_holdout_unused=list(engine.FINAL),
            scope='One-seed hit-return acquisition development screen; not score-only mastery or complete M1B')
        atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare(args.directory)
    else:supervise(args.directory)
