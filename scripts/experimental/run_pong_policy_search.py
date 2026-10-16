"""Registered score-only smooth linear policy search; frozen camera perception."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from contextlib import redirect_stdout
from dataclasses import asdict
import fcntl,hashlib,io,json,multiprocessing as mp,os,sqlite3,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy import sparse
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import run_pong_development as base
from policy_search_math import (SEARCH_CONFIG,SearchTransform,fit_transform,function_rms,
                                sample_directions,generation_spec,mirrored_update)
from fly_connectome.control.session import source_signature,MODEL_HASH
from fly_connectome.control.durable import runtime_signature
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.vision import VisualStateEncoder

ROOT=Path(__file__).resolve().parents[2]
WORKER=Path(base.__file__);MATH=Path(__file__).with_name('policy_search_math.py')
CACHE=Path('/workspace/m1b-experiments/ORACLE-DIAGNOSTIC-readout-capacity-20261016/generic-calibration-features.npz')
CAP_REFERENCE=Path('/workspace/m1b-experiments/pong-alpha-hold-20261016-r1')
CONTROL_REFERENCE=Path('/workspace/m1b-experiments/pong-exposure-sharing-20261015-r1')
OPTIMIZER_SEED=1873926119
TRAINING_SEED=172900127
GENERATIONS=8;TRAIN_CAMERAS=60000;TRAIN_SECONDS=1440.;TOTAL_SECONDS=1800.
DEV=904271833;FINAL=(308713321,1203078761,1983461927)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_hash(value):return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
def atomic_json(path,value):base.atomic_json(Path(path),value)
def immutable_array(path,value):
    path=Path(path)
    if path.exists():
        if not np.array_equal(np.load(path,allow_pickle=False),value):raise ValueError('immutable array differs: '+str(path))
    else:base.atomic_weights(path,value)
    return dict(shape=list(value.shape),weight_hash=array_hash(value))
def weights_file(directory,name,value):
    path=directory/name;metadata=immutable_array(path,value)
    meta=path.with_suffix('.json')
    if meta.exists() and json.loads(meta.read_text())!=metadata:raise ValueError('immutable weight metadata differs')
    atomic_json(meta,metadata)
    return metadata

def read_calibration(path):
    with np.load(path,allow_pickle=False) as a:
        if set(a.files)!={'indices','values','indptr'}:raise ValueError('only generic sparse input arrays allowed')
        result=sparse.csr_matrix((a['values'].astype(np.float64),a['indices'],a['indptr']),shape=(1024,4389))
    if not np.isfinite(result.data).all():raise ValueError('nonfinite calibration')
    return result


def pinned_sources():
    return dict(source=source_signature(),runtime=runtime_signature(),model_hash=MODEL_HASH,
        driver_hash=sha(WORKER),coordinator_hash=sha(__file__),math_hash=sha(MATH),
        calibration_source_hash=sha(CACHE),cap_hash=sha(CAP_REFERENCE/'radial-cap.json'),
        reference_report_hashes={n:sha(CONTROL_REFERENCE/f'eval_{n}.json') for n in ('initial','up','random','noop')},
        baseline_proof_hash=sha(CAP_REFERENCE/'baseline-projection-proof.json'),
        search_config=SEARCH_CONFIG,optimizer_seed=OPTIMIZER_SEED,training_seed=TRAINING_SEED,
        max_generations=GENERATIONS,training_camera_cap=TRAIN_CAMERAS,
        training_wall_seconds=TRAIN_SECONDS,total_wall_seconds=TOTAL_SECONDS,
        protocol='docs/reconstruction/2026-10-16-score-only-policy-search-protocol.md',
        protocol_hash=sha(ROOT/'docs/reconstruction/2026-10-16-score-only-policy-search-protocol.md'))


def prepare(directory):
    directory.mkdir(parents=True,exist_ok=True);path=directory/'manifest.json';expected=pinned_sources()
    if path.exists():
        manifest=json.loads(path.read_text())
        if any(manifest[k]!=v for k,v in expected.items()):raise ValueError('changed declared source/runtime/configuration')
    else:
        proof=json.loads((CAP_REFERENCE/'baseline-projection-proof.json').read_text())
        reference_manifest=json.loads((CONTROL_REFERENCE/'manifest.json').read_text())
        if (proof['source']!=expected['source'] or proof['reference_source']!=reference_manifest['source']
                or reference_manifest['runtime']!=expected['runtime'] or not all(p['exact'] for p in proof['proof'])):
            raise ValueError('baseline proof/source/runtime differs')
        manifest=dict(**expected,started_wall_ns=time.time_ns(),jobs=[],
            purpose='score-only policy search; training fitness is not fresh competence evaluation',
            opponent_speed=.25,hold=1,workers=3,long_bank_entries=0,context_fallback_steps=4,
            reserved_final_seeds=list(FINAL),old_calibration_observations_reused=1024,new_calibration_observations=0)
        atomic_json(path,manifest)
    if not (directory/'radial-cap.json').exists():(directory/'radial-cap.json').write_bytes((CAP_REFERENCE/'radial-cap.json').read_bytes())
    if sha(directory/'radial-cap.json')!=expected['cap_hash']:raise ValueError('cap artifact differs')
    transform_path=directory/'search-transform.npy'
    raw=read_calibration(CACHE)
    if transform_path.exists() and (directory/'transform-fit.json').exists():
        transform=SearchTransform(np.load(transform_path,allow_pickle=False),json.loads((directory/'transform-fit.json').read_text()))
        if array_hash(transform.matrix)!=transform.diagnostics['matrix_hash']:raise ValueError('transform differs')
        calibration=np.asarray(raw@transform.matrix.T)
    else:
        start=time.monotonic();encoder=VisualStateEncoder(64,64,.02)
        transform,calibration=fit_transform(raw,encoder.layout,encoder.dimension)
        transform.diagnostics.update(matrix_hash=array_hash(transform.matrix),calibration_source_hash=sha(CACHE),
            fit_wall_seconds=time.monotonic()-start,new_observations=0,old_calibration_rows_reused=1024)
        immutable_array(transform_path,transform.matrix);atomic_json(directory/'transform-fit.json',transform.diagnostics)
    transform_pins=dict(transform_array_hash=array_hash(transform.matrix),transform_fit_hash=sha(directory/'transform-fit.json'))
    if 'transform_array_hash' in manifest:
        if any(manifest[k]!=v for k,v in transform_pins.items()):raise ValueError('fixed transform pins differ')
    else:
        manifest.update(transform_pins);atomic_json(path,manifest)
    state_path=directory/'search-state.json'
    if not state_path.exists():
        rng=np.random.default_rng(OPTIMIZER_SEED)
        theta=SEARCH_CONFIG['initial_scale']*sample_directions(rng,calibration,1)[0]
        immutable_array(directory/'theta_at0.npy',theta);weights_file(directory,'policy_at0.npy',transform.mapped_weights(theta))
        atomic_json(state_path,dict(completed_generations=0,theta_file='theta_at0.npy',theta_hash=array_hash(theta),
            rng_state=rng.bit_generator.state,initial_policy_hash=sha(directory/'policy_at0.npy')))
    state=json.loads(state_path.read_text());theta=np.load(directory/state['theta_file'],allow_pickle=False)
    if array_hash(theta)!=state['theta_hash']:raise ValueError('optimizer checkpoint differs')
    weights_file(directory,f"policy_at{state['completed_generations']}.npy",transform.mapped_weights(theta))
    return manifest,transform,calibration,state,theta


def register_jobs(directory,jobs):
    path=directory/'manifest.json';manifest=json.loads(path.read_text());known={j['id']:j for j in manifest['jobs']}
    for job in jobs:
        if job['id'] in known and known[job['id']]!=job:raise ValueError('registered rollout differs')
        if job['id'] not in known:manifest['jobs'].append(job)
    atomic_json(path,manifest)


def rollout_job(name,seed,weights,purpose,episodes=1):
    return dict(id=name,kind='evaluate',purpose=purpose,
        config=asdict(PongEvaluationConfig(seed,episodes=episodes,opponent_speed=.25)),
        weights_from=weights,scaling_from='radial-cap.json',compact_reports=True)


def prepare_generation(directory,transform,calibration,state,theta):
    generation=state['completed_generations'];directions,candidates,serves,after=generation_spec(theta,state['rng_state'],generation,calibration,training_seed=TRAINING_SEED)
    dirname=f'g{generation:02d}';jobs=[]
    immutable_array(directory/(dirname+'-directions.npy'),directions)
    for direction in range(8):
        for sign in range(2):
            label=f'{dirname}_d{direction}_{"plus" if sign==0 else "minus"}'
            filename=label+'.npy';weights_file(directory,filename,transform.mapped_weights(candidates[direction,sign]))
            for serve,seed in enumerate(serves):jobs.append(rollout_job(f'{label}_s{serve}',seed,filename,'training_fitness'))
    definition=dict(generation=generation,input_theta_hash=array_hash(theta),input_rng_state=state['rng_state'],
        directions_hash=array_hash(directions),training_serve_seeds=serves,rng_state_after_draw=after,jobs=jobs)
    path=directory/(dirname+'-definition.json')
    if path.exists() and json.loads(path.read_text())!=definition:raise ValueError('pending generation differs')
    atomic_json(path,definition);register_jobs(directory,jobs)
    return definition,directions


def sample_count(directory,name):
    path=directory/(name+'.db')
    if not path.exists():return 0
    with sqlite3.connect(path) as connection:return connection.execute('SELECT count(*) FROM replay_samples').fetchone()[0]

def read_report(directory,name):
    path=directory/(name+'.json')
    return json.loads(path.read_text()) if path.exists() else None


def journal_hash(path):
    """Pin every semantic journal table; timing may grow on verified read opens."""
    result=hashlib.sha256()
    with sqlite3.connect(path) as connection:
        for table,order in [('replay_meta','id'),('replay_artifacts','hash'),('replay_samples','seq'),('replay_active','id')]:
            result.update(table.encode()+b'\0')
            for row in connection.execute(f'SELECT * FROM {table} ORDER BY {order}'):
                for value in row:
                    raw=value if isinstance(value,bytes) else str(value).encode()
                    result.update(type(value).__name__.encode()+b':'+str(len(raw)).encode()+b':'+raw)
    return result.hexdigest()


def pin_completion(directory,name):
    report=read_report(directory,name)
    if not report or not report['complete']:return
    value=dict(report_hash=sha(directory/(name+'.json')),journal_hash=journal_hash(directory/(name+'.db')))
    atomic_json(directory/(name+'-completion.json'),value)


def complete_report(directory,name):
    path=directory/(name+'-completion.json')
    report=read_report(directory,name)
    if not path.exists():return None
    value=json.loads(path.read_text())
    if not report or not report['complete'] or value['report_hash']!=sha(directory/(name+'.json')) or value['journal_hash']!=journal_hash(directory/(name+'.db')):
        raise ValueError('completed rollout artifacts changed')
    return report


def rollout_task(directory,name,camera_grant,work_seconds):
    """Persistent spawned workers reuse imports across individually durable serves."""
    directory=Path(directory);started=time.monotonic()
    try:
        with redirect_stdout(io.StringIO()):
            if camera_grant==0:
                if not (read_report(directory,name) or {}).get('complete'):raise ValueError('zero grant requires completed recovery verification')
                report=base.worker(directory,name,work_seconds,emit=False)
                result={k:report[k] for k in ('transitions','new_transitions','replayed_transitions','worker_wall_seconds')}
            else:result=base.multiple_chunks(directory,name,work_seconds,min(5,camera_grant//200),warm_session=True)
        return dict(job=name,ok=True,result=result,wall_seconds=time.monotonic()-started)
    except BaseException:
        return dict(job=name,ok=False,error=traceback.format_exc(),wall_seconds=time.monotonic()-started)


def execute_jobs(directory,jobs,pool,*,deadline,camera_cap,initial_consumed=0):
    """Reserve additional camera work BEFORE dispatch, including concurrent grants."""
    pending=[j for j in jobs if complete_report(directory,j['id']) is None]
    settled=initial_consumed;active={};failed=False
    while pending or active:
        available=camera_cap-settled-sum(row['grant'] for row in active.values())
        while pending and len(active)<3 and deadline-time.time()>25 and not failed:
            job=pending[0]
            verification_only=bool((read_report(directory,job['id']) or {}).get('complete'))
            if available<200 and not verification_only:break
            pending.pop(0);before=sample_count(directory,job['id'])
            grant=0 if verification_only else min(1000,available//200*200)
            future=pool.submit(rollout_task,str(directory),job['id'],grant,min(160.,deadline-time.time()-5))
            active[future]=dict(job=job,before=before,grant=grant);available-=grant
        if not active:break
        done,_=wait(active,timeout=min(1.,max(.001,deadline-time.time())),return_when=FIRST_COMPLETED)
        for future in done:
            reservation=active.pop(future)
            try:cost=future.result()
            except BaseException:cost=dict(job=reservation['job']['id'],ok=False,error=traceback.format_exc())
            after=sample_count(directory,reservation['job']['id']);new=after-reservation['before']
            if not 0<=new<=reservation['grant']:raise ValueError('camera reservation exceeded')
            settled+=new;cost.update(new_committed_cameras=new,camera_grant=reservation['grant'])
            with (directory/'rollout-costs.jsonl').open('a') as stream:
                stream.write(json.dumps(cost)+'\n');stream.flush();os.fsync(stream.fileno())
            if not cost['ok']:failed=True
            else:pin_completion(directory,reservation['job']['id'])
            report=read_report(directory,reservation['job']['id'])
            if cost['ok'] and not (report or {}).get('complete'):pending.insert(0,reservation['job'])
    complete=all(complete_report(directory,j['id']) is not None for j in jobs)
    return dict(complete=complete,failed=failed,committed_cameras=settled,unfinished=[j['id'] for j in jobs if not (read_report(directory,j['id']) or {}).get('complete')])


def finish_generation(directory,definition,directions,theta,calibration):
    returns=np.zeros((8,2));reports=[]
    for index,job in enumerate(definition['jobs']):
        report=complete_report(directory,job['id'])
        if not report or not report['complete'] or report['episodes']!=1:raise ValueError('partial generation cannot update')
        reward=report['total_reward']
        if reward not in (-1.,0.,1.):raise ValueError('fitness must be terminal score only')
        returns[index//8,(index//4)%2]+=reward/4
        reports.append(dict(job=job['id'],terminal_score=reward,transitions=report['transitions']))
    proposed,diagnostics=mirrored_update(theta,directions,returns,calibration,top=4,step_size=SEARCH_CONFIG['step_size'])
    generation=definition['generation'];count=generation+1
    immutable_array(directory/f'theta_at{count}.npy',proposed)
    weights_file(directory,f'policy_at{count}.npy',SearchTransform(np.load(directory/'search-transform.npy',allow_pickle=False),{}).mapped_weights(proposed))
    result=dict(generation=generation,returns=returns.tolist(),optimizer=diagnostics,rollouts=reports,
        new_theta_hash=array_hash(proposed),fitness='mean of four terminal scores; truncation0; no hit/label shaping')
    path=directory/f'g{generation:02d}-result.json'
    if path.exists() and json.loads(path.read_text())!=result:raise ValueError('completed generation differs')
    atomic_json(path,result)
    state=dict(completed_generations=count,theta_file=f'theta_at{count}.npy',theta_hash=array_hash(proposed),
        rng_state=definition['rng_state_after_draw'],initial_policy_hash=sha(directory/'policy_at0.npy'))
    atomic_json(directory/'search-state.json',state)
    return state,proposed,result


def counts(directory,manifest):
    result=dict(training_fitness=0,fresh_evaluation=0)
    for job in manifest['jobs']:result[job['purpose']]+=sample_count(directory,job['id'])
    return result


def supervise(directory):
    torch.set_num_threads(1);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest,transform,calibration,state,theta=prepare(directory)
        started=manifest['started_wall_ns']/1e9;training_deadline=started+TRAIN_SECONDS;deadline=started+TOTAL_SECONDS
        generation_results=[];training_stop=None
        with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
            endpoint_path=directory/'training-endpoint.json'
            prior_costs=[json.loads(line) for line in (directory/'rollout-costs.jsonl').read_text().splitlines()] if (directory/'rollout-costs.jsonl').exists() else []
            if endpoint_path.exists():
                frozen=json.loads(endpoint_path.read_text())
                if frozen['completed_generations']!=state['completed_generations'] or frozen['theta_hash']!=state['theta_hash']:
                    raise ValueError('frozen endpoint differs from optimizer checkpoint')
                training_stop=frozen['training_stop']
            elif any(not c['ok'] for c in prior_costs):training_stop='prior logged rollout failure'
            while not endpoint_path.exists() and training_stop is None and state['completed_generations']<GENERATIONS:
                definition,directions=prepare_generation(directory,transform,calibration,state,theta)
                current=counts(directory,json.loads((directory/'manifest.json').read_text()))['training_fitness']
                if current>=TRAIN_CAMERAS or training_deadline-time.time()<=25:
                    training_stop='registered camera/wall budget';break
                progress=execute_jobs(directory,definition['jobs'],pool,deadline=training_deadline,camera_cap=TRAIN_CAMERAS,initial_consumed=current)
                if not progress['complete']:
                    training_stop='rollout failure' if progress['failed'] else 'registered camera/wall budget'
                    atomic_json(directory/'incomplete-generation.json',dict(generation=state['completed_generations'],**progress));break
                state,theta,result=finish_generation(directory,definition,directions,theta,calibration)
                generation_results.append(result)
                print(json.dumps(dict(completed_generations=state['completed_generations'],training_cameras=progress['committed_cameras'],returns=result['returns'],optimizer=result['optimizer'])),flush=True)
            if not endpoint_path.exists():
                atomic_json(endpoint_path,dict(completed_generations=state['completed_generations'],theta_hash=state['theta_hash'],
                    training_stop=training_stop or 'eight complete generations',frozen_wall_ns=time.time_ns()))
            evaluation_jobs=[]
            if state['completed_generations']>=2:
                evaluation_jobs=[rollout_job('eval_endpoint',DEV,f"policy_at{state['completed_generations']}.npy",'fresh_evaluation',24),
                                 rollout_job('eval_initial_restricted',DEV,'policy_at0.npy','fresh_evaluation',24)]
                register_jobs(directory,evaluation_jobs)
                evaluation=execute_jobs(directory,evaluation_jobs,pool,deadline=deadline,camera_cap=48000,
                    initial_consumed=counts(directory,json.loads((directory/'manifest.json').read_text()))['fresh_evaluation'])
            else:evaluation=dict(complete=False,reason='fewer than two full generations')
        final_manifest=json.loads((directory/'manifest.json').read_text());actual=counts(directory,final_manifest)
        assert actual['training_fitness']<=TRAIN_CAMERAS and actual['fresh_evaluation']<=48000
        controls=[json.loads((CONTROL_REFERENCE/f'eval_{n}.json').read_text()) for n in ('initial','up','random','noop')]
        endpoint=complete_report(directory,'eval_endpoint');initial=complete_report(directory,'eval_initial_restricted')
        matched=controls+([initial] if initial and initial['complete'] else [])
        gate=dict(total=max(14,max(r['wins'] for r in matched)+4),own_hit=max(6,max(r['own_hit_followed_wins'] for r in matched)+4))
        qualifies=bool(evaluation['complete'] and endpoint and endpoint['complete'] and endpoint['wins']>=gate['total'] and endpoint['own_hit_followed_wins']>=gate['own_hit'])
        camera_images=0;replays=0
        for job in final_manifest['jobs']:
            report=read_report(directory,job['id'])
            db=directory/(job['id']+'.db')
            if db.exists():
                with sqlite3.connect(db) as connection:
                    episodes=set();number=0
                    for (raw,) in connection.execute('SELECT record FROM replay_samples ORDER BY seq'):
                        episodes.add(json.loads(raw)['episode']);number+=1
                    camera_images+=number+len(episodes)
        costs=[json.loads(line) for line in (directory/'rollout-costs.jsonl').read_text().splitlines()] if (directory/'rollout-costs.jsonl').exists() else []
        replays=sum(c.get('result',{}).get('replayed_transitions',0) for c in costs)
        summary=dict(status='complete' if evaluation['complete'] else 'partial',counts=actual,images_including_resets=camera_images,
            diagnostic_replayed_camera_steps=replays,uncommitted_failed_observation_tails_not_claimed_zero=True,completed_generations=state['completed_generations'],selected_checkpoint=f"policy_at{state['completed_generations']}.npy",
            selection='last complete generation at prespecified budget end; never best-checkpoint',training_stop=training_stop,
            endpoint=endpoint,initial_restricted=initial,controls=controls,gate=gate,qualifies=qualifies,
            no_automatic_replication=True,wall_seconds=time.time()-started,
            summed_task_wall_seconds=sum(c.get('wall_seconds',0) for c in costs),transform_diagnostics=transform.diagnostics,
            final_seeds_unused=list(FINAL),scope='opponent.25; score-only policy search; generic restricted linear class')
        atomic_json(directory/'summary.json',summary)
        print(json.dumps({k:summary[k] for k in ('status','counts','completed_generations','gate','qualifies','wall_seconds')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        torch.set_num_threads(1);m,t,z,s,w=prepare(args.directory);print(json.dumps(t.diagnostics),flush=True)
    else:supervise(args.directory)
