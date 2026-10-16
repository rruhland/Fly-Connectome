"""Fixed stationary-to-.25 score-only curriculum; original policy-search engine."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import fcntl,json,multiprocessing as mp,sqlite3,time
from pathlib import Path
import numpy as np
import torch
import run_pong_policy_search_continuation as engine
from policy_search_math import SEARCH_CONFIG,SearchTransform,sample_directions,generation_spec
from fly_connectome.control.session import source_signature,MODEL_HASH
from fly_connectome.control.durable import runtime_signature
from fly_connectome.control.pong_evaluation import PongEvaluationConfig

ROOT=Path(__file__).resolve().parents[2]
ORIGINAL=engine.LINEAGE;CONTROL=engine.CONTROL_REFERENCE;CACHE=engine.CACHE
BRIDGE=Path('/workspace/m1b-experiments/curriculum-prelaunch-20261016/source-bridge.json')
PRIOR_BRIDGE=Path('/workspace/m1b-experiments/pong-score-policy-search-continuation-20261016-r1/source-bridge.json')
PROTOCOL=ROOT/'docs/reconstruction/2026-10-16-stationary-curriculum-protocol.md'
PHASES=(dict(name='phase0-stationary',opponent_speed=0.,generations=6,camera_cap=30000,wall_seconds=480.),
        dict(name='phase1-target',opponent_speed=.25,generations=8,camera_cap=60000,wall_seconds=960.))
TOTAL_SECONDS=1800.;TRAINING_SEED=172900127;OPTIMIZER_SEED=1873926119
DEV=engine.DEV;FINAL=engine.FINAL
sha=engine.sha;array_hash=engine.array_hash;atomic_json=engine.atomic_json
immutable_array=engine.immutable_array;weights_file=engine.weights_file


def pins():
    return dict(source=source_signature(),runtime=runtime_signature(),model_hash=MODEL_HASH,
        driver_hash=sha(engine.WORKER),engine_hash=sha(engine.__file__),coordinator_hash=sha(__file__),
        math_hash=sha(engine.MATH),protocol_hash=sha(PROTOCOL),source_bridge_hash=sha(BRIDGE),
        prior_source_bridge_hash=sha(PRIOR_BRIDGE),calibration_hash=sha(CACHE),
        transform_hash=sha(ORIGINAL/'search-transform.npy'),transform_fit_hash=sha(ORIGINAL/'transform-fit.json'),
        cap_hash=sha(ORIGINAL/'radial-cap.json'),original_initial_report_hash=sha(ORIGINAL/'eval_initial_restricted.json'),
        original_initial_weights_hash=sha(ORIGINAL/'policy_at0.npy'),original_first_definition_hash=sha(ORIGINAL/'g00-definition.json'),
        control_report_hashes={n:sha(CONTROL/f'eval_{n}.json') for n in ('initial','up','random','noop')},
        search_config=SEARCH_CONFIG,phases=list(PHASES),total_wall_seconds=TOTAL_SECONDS,workers=4,
        training_seed=TRAINING_SEED,optimizer_seed=OPTIMIZER_SEED,long_bank_entries=0,context_fallback_steps=4,
        reserved_final_seeds=list(FINAL),new_calibration_observations=0,reused_calibration_rows=1024)


def prepare(root):
    root.mkdir(parents=True,exist_ok=True);expected=pins();path=root/'manifest.json'
    bridge=json.loads(BRIDGE.read_text());prior=json.loads(PRIOR_BRIDGE.read_text())
    ancestor=json.loads((ORIGINAL/'manifest.json').read_text())
    if (ancestor['runtime']!=expected['runtime'] or ancestor['math_hash']!=expected['math_hash']
            or ancestor['search_config']!=expected['search_config']
            or ancestor['calibration_source_hash']!=expected['calibration_hash']
            or bridge['source']!=expected['source'] or bridge['reference_source']!=prior['source']
            or bridge['prior_source_bridge_hash']!=expected['prior_source_bridge_hash']
            or prior['reference_source']!=ancestor['source']
            or not bridge['only_nonnegative_validation_change']
            or not all(x['exact'] for x in bridge['positive_speed_action_physics_projection'])):
        raise ValueError('baseline source/projection chain differs')
    if path.exists():
        manifest=json.loads(path.read_text())
        if any(manifest[k]!=v for k,v in expected.items()):raise ValueError('changed registered source/runtime/configuration')
    else:
        manifest=dict(**expected,started_wall_ns=time.time_ns(),jobs=[],purpose='fixed stationary-to-.25 score-only curriculum')
        atomic_json(path,manifest)
    for name in ('search-transform.npy','transform-fit.json','radial-cap.json'):
        target=root/name
        if not target.exists():target.write_bytes((ORIGINAL/name).read_bytes())
        if sha(target)!=sha(ORIGINAL/name):raise ValueError('fixed transform/cap differs')
    transform=SearchTransform(np.load(root/'search-transform.npy'),json.loads((root/'transform-fit.json').read_text()))
    z=np.asarray(engine.read_calibration(CACHE)@transform.matrix.T)
    rng=np.random.default_rng(OPTIMIZER_SEED);theta=.1*sample_directions(rng,z,1)[0]
    immutable_array(root/'theta_initial.npy',theta);weights_file(root,'policy_initial.npy',transform.mapped_weights(theta))
    if not np.array_equal(np.load(root/'policy_initial.npy'),np.load(ORIGINAL/'policy_at0.npy')):
        raise ValueError('original initial readout differs')
    return manifest,transform,z,theta,rng.bit_generator.state


def initialize_phase(root,spec,theta,rng,global_offset,started_wall_ns):
    directory=root/spec['name'];directory.mkdir(exist_ok=True)
    declared=pins();declared.update(phase=spec,global_generation_offset=global_offset,phase_started_wall_ns=started_wall_ns,
        input_theta_hash=array_hash(theta),input_rng_state=rng)
    path=directory/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if any(manifest[k]!=v for k,v in declared.items()):raise ValueError('phase lineage/budget/source differs')
    else:atomic_json(path,dict(**declared,jobs=[]))
    for name in ('search-transform.npy','radial-cap.json'):
        target=directory/name
        if not target.exists():target.write_bytes((root/name).read_bytes())
        if sha(target)!=sha(root/name):raise ValueError('phase transform/cap differs')
    immutable_array(directory/'theta_at0.npy',theta)
    weights_file(directory,'policy_at0.npy',theta@np.load(root/'search-transform.npy'))
    state_path=directory/'search-state.json'
    if not state_path.exists():atomic_json(state_path,dict(completed_generations=0,theta_file='theta_at0.npy',theta_hash=array_hash(theta),rng_state=rng,initial_policy_hash=sha(directory/'policy_at0.npy')))
    state=json.loads(state_path.read_text());current=np.load(directory/state['theta_file'])
    if array_hash(current)!=state['theta_hash']:raise ValueError('phase optimizer checkpoint differs')
    return directory,state,current


def prepare_generation(directory,transform,z,state,theta,global_offset,speed):
    local=state['completed_generations'];global_index=global_offset+local
    directions,candidates,serves,after=generation_spec(theta,state['rng_state'],global_index,z,training_seed=TRAINING_SEED)
    name=f'g{local:02d}';immutable_array(directory/(name+'-directions.npy'),directions);jobs=[]
    for direction in range(8):
        for sign in range(2):
            label=f'{name}_d{direction}_{"plus" if sign==0 else "minus"}';filename=label+'.npy'
            weights_file(directory,filename,transform.mapped_weights(candidates[direction,sign]))
            for serve,seed in enumerate(serves):
                job=engine.rollout_job(f'{label}_s{serve}',seed,filename,'training_fitness')
                job['config']['opponent_speed']=speed;jobs.append(job)
    definition=dict(generation=local,global_generation=global_index,input_theta_hash=array_hash(theta),input_rng_state=state['rng_state'],directions_hash=array_hash(directions),training_serve_seeds=serves,rng_state_after_draw=after,jobs=jobs)
    path=directory/(name+'-definition.json')
    if path.exists() and json.loads(path.read_text())!=definition:raise ValueError('pending phase population differs')
    atomic_json(path,definition);engine.register_jobs(directory,jobs)
    return definition,directions


def next_boundary(directory,state,global_offset):
    """An unused draw advances RNG/index; its partial fitness never transfers."""
    count=state['completed_generations'];path=directory/f'g{count:02d}-definition.json'
    if path.exists():
        pending=json.loads(path.read_text())
        if (pending['input_rng_state']!=state['rng_state'] or pending['input_theta_hash']!=state['theta_hash']
                or pending['global_generation']!=global_offset+count):raise ValueError('pending transition draw differs')
        return pending['rng_state_after_draw'],global_offset+count+1,True
    return state['rng_state'],global_offset+count,False


def run_phase(directory,spec,transform,z,state,theta,global_offset,pool,deadline):
    marker=directory/'training-endpoint.json'
    if marker.exists():
        frozen=json.loads(marker.read_text())
        rng,index,skipped=next_boundary(directory,state,global_offset)
        if (frozen['completed_generations']!=state['completed_generations'] or frozen['theta_hash']!=state['theta_hash']
                or frozen['next_rng_state']!=rng or frozen['next_global_generation']!=index or frozen['unused_draw_skipped']!=skipped):raise ValueError('frozen phase endpoint differs')
        return frozen,theta
    costs=[json.loads(x) for x in (directory/'rollout-costs.jsonl').read_text().splitlines()] if (directory/'rollout-costs.jsonl').exists() else []
    failed=any(not c['ok'] for c in costs);stop='prior task failure' if failed else None
    while not failed and state['completed_generations']<spec['generations']:
        definition,directions=prepare_generation(directory,transform,z,state,theta,global_offset,spec['opponent_speed'])
        current=engine.counts(directory,json.loads((directory/'manifest.json').read_text()))['training_fitness']
        if current>=spec['camera_cap'] or deadline-time.time()<=25:stop='registered phase camera/wall budget';break
        progress=engine.execute_jobs(directory,definition['jobs'],pool,deadline=deadline,camera_cap=spec['camera_cap'],initial_consumed=current)
        if not progress['complete']:
            failed=progress['failed'];stop='task failure' if failed else 'registered phase camera/wall budget'
            atomic_json(directory/'incomplete-generation.json',dict(generation=state['completed_generations'],**progress));break
        state,theta,result=engine.finish_generation(directory,definition,directions,theta,z)
        print(json.dumps(dict(phase=spec['name'],completed_generations=state['completed_generations'],training_cameras=progress['committed_cameras'],returns=result['returns'])),flush=True)
    rng,index,skipped=next_boundary(directory,state,global_offset)
    observed=engine.counts(directory,json.loads((directory/'manifest.json').read_text()))['training_fitness']
    used=sum(sum(r['transitions'] for r in json.loads((directory/f'g{g:02d}-result.json').read_text())['rollouts']) for g in range(state['completed_generations']))
    frozen=dict(completed_generations=state['completed_generations'],theta_hash=state['theta_hash'],theta_file=state['theta_file'],next_rng_state=rng,next_global_generation=index,unused_draw_skipped=skipped,
        failed=failed,valid=not failed and state['completed_generations']>=2,training_stop=stop or 'registered complete generations',observed_cameras=observed,used_cameras=used,unused_cameras=observed-used,frozen_wall_ns=time.time_ns())
    atomic_json(marker,frozen);return frozen,theta


def summarize_costs(root):
    totals=dict(training_fitness=0,fresh_evaluation=0,images_including_resets=0,replayed_cameras=0,summed_task_wall_seconds=0.)
    for directory in [root,*[root/s['name'] for s in PHASES]]:
        if not (directory/'manifest.json').exists():continue
        manifest=json.loads((directory/'manifest.json').read_text())
        for job in manifest['jobs']:
            number=engine.sample_count(directory,job['id']);totals[job['purpose']]+=number
            if number:
                with sqlite3.connect(directory/(job['id']+'.db')) as c:episodes={json.loads(r[0])['episode'] for r in c.execute('SELECT record FROM replay_samples ORDER BY seq')}
                totals['images_including_resets']+=number+len(episodes)
        costs=[json.loads(x) for x in (directory/'rollout-costs.jsonl').read_text().splitlines()] if (directory/'rollout-costs.jsonl').exists() else []
        totals['summed_task_wall_seconds']+=sum(c.get('wall_seconds',0.) for c in costs)
        totals['replayed_cameras']+=sum(c.get('result',{}).get('replayed_transitions',0) for c in costs)
    return totals


def supervise(root):
    torch.set_num_threads(1);root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest,transform,z,theta,rng=prepare(root);start=manifest['started_wall_ns'];index=0;anchor=start;phases=[]
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
            for spec in PHASES:
                directory,state,current=initialize_phase(root,spec,theta,rng,index,anchor)
                frozen,theta=run_phase(directory,spec,transform,z,state,current,index,pool,
                    deadline=min(anchor/1e9+spec['wall_seconds'],start/1e9+1440.))
                phases.append(dict(phase=spec['name'],**frozen))
                if not frozen['valid']:break
                rng,index,anchor=frozen['next_rng_state'],frozen['next_global_generation'],frozen['frozen_wall_ns']
            evaluation=dict(complete=False,reason='invalid training phase')
            if len(phases)==2 and all(s['valid'] for s in phases):
                weights_file(root,'policy_endpoint.npy',transform.mapped_weights(theta))
                job=engine.rollout_job('eval_endpoint',DEV,'policy_endpoint.npy','fresh_evaluation',24)
                engine.register_jobs(root,[job])
                evaluation=engine.execute_jobs(root,[job],pool,deadline=start/1e9+TOTAL_SECONDS,camera_cap=24000,
                    initial_consumed=engine.counts(root,json.loads((root/'manifest.json').read_text()))['fresh_evaluation'])
        endpoint=engine.complete_report(root,'eval_endpoint')
        original=json.loads((ORIGINAL/'eval_initial_restricted.json').read_text())
        controls=[json.loads((CONTROL/f'eval_{n}.json').read_text()) for n in ('initial','up','random','noop')]
        gate=engine.control_gate([original,*controls]);cost=summarize_costs(root)
        assert cost['training_fitness']<=90000 and cost['fresh_evaluation']<=24000
        qualifies=bool(evaluation['complete'] and endpoint and endpoint['wins']>=gate['total'] and endpoint['own_hit_followed_wins']>=gate['own_hit'])
        result=dict(status='complete' if evaluation['complete'] else 'partial',phases=phases,counts=cost,
            endpoint=endpoint,original_initial=original,controls=controls,gate=gate,qualifies=qualifies,
            selection='last full target-phase update; original from-zero control; never best or phase2-initial control',
            wall_seconds=time.time()-start/1e9,final_seeds_unused=list(FINAL),no_automatic_replication=True,
            scope='stationary training curriculum transfers to unchanged .25; no original.45 or forecast-benefit claim',
            uncommitted_failed_observation_tails_not_claimed_zero=True)
        atomic_json(root/'summary.json',result);print(json.dumps(dict(status=result['status'],counts=cost,gate=gate,qualifies=qualifies)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True,type=Path);parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:
        torch.set_num_threads(1);prepare(args.directory)
    else:supervise(args.directory)
