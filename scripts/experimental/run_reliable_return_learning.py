"""Bounded hit-only return learning at hold1/10 against the .65 opponent."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import asdict
import fcntl,json,multiprocessing as mp,os,sqlite3,sys,time
from pathlib import Path
import numpy as np
import run_dct_sarsa_pong as base
import run_pong_policy_search_continuation as rollout
import run_pong_development as worker
from fly_connectome.control.session import source_signature,MODEL_HASH
from fly_connectome.control.durable import runtime_signature
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.projection import FixedPolicyProjection
from fly_connectome.control.rewards import REWARD_VERSION
from return_metrics import replay_records,aggregate,paired_evidence

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'docs/reconstruction/2026-10-16-reliable-return-learning-protocol.md'
OPPONENT=.65;TRAIN_SECONDS=720.;TOTAL_SECONDS=1320.;SERVES=24
sha=base.sha

def serve_schedule():
    values=np.random.default_rng(np.random.SeedSequence([104729113,29071])).integers(1,2147483647,size=SERVES)
    result=[int(x) for x in values]
    if len(set(result))!=SERVES or set(result)&set(base.FINAL):raise ValueError('declared fresh development stream collision')
    return result

def declared_jobs():
    learning=[dict(id=f'learn_hold{hold}',kind='learn',seed=172900127,alpha=.001,lam=.995,
        opponent_speed=OPPONENT,action_hold_frames=hold,transitions=12000,checkpoints=[12000],
        projection_from='projection.json',compact_reports=True,max_abs_q=1000.,
        reward_shaping=dict(version=REWARD_VERSION,score_coefficient=0.,own_hit_bonus=1.,survival_per_second=0.)) for hold in (1,10)]
    evaluation=[]
    for policy,hold in [('hold1',1),('hold10',10),('initial',1),('up',1),('noop',1),('random1',1),('random10',10)]:
        trained=policy in ('hold1','hold10')
        for i,seed in enumerate(serve_schedule()):
            kind='trained' if trained else ('random' if policy.startswith('random') else policy)
            action_seed=int(np.random.SeedSequence([447201113,104729113,i]).generate_state(1)[0]) if kind=='random' else None
            config=asdict(PongEvaluationConfig(seed,episodes=1,max_episode_ticks=1200,
                policy_kind=kind,action_seed=action_seed,skip_baseline_perception=not trained,
                opponent_speed=OPPONENT,action_hold_frames=hold))
            evaluation.append(dict(id=f'eval_{policy}_s{i:02d}',kind='evaluate',policy=policy,serve_index=i,
                config=config,weights_from=f'learn_{policy}_at12000.npy' if trained else None,
                scaling_from=f'learn_{policy}_at12000.cap.json' if trained else None,compact_reports=True))
    return learning,evaluation

def prepare(directory):
    directory.mkdir(parents=True,exist_ok=True)
    descriptor=json.loads((directory/'projection.json').read_text());projection=FixedPolicyProjection(4389,descriptor)
    projection.validate_embedded()
    if descriptor['file_sha256']!=sha(base.TRANSFORM) or descriptor['max_norm2']!=256. or descriptor['shape']!=[214,4389]:
        raise ValueError('exact frozen DCT and cap required')
    learning,evaluation=declared_jobs()
    expected=dict(source=source_signature(),runtime=runtime_signature(),model_hash=MODEL_HASH,jobs=learning+evaluation,
        driver_hash=sha(worker.__file__),coordinator_hash=sha(__file__),completion_helper_hash=sha(base.__file__),
        rollout_helper_hash=sha(rollout.__file__),metrics_hash=sha(Path(__file__).with_name('return_metrics.py')),
        protocol_hash=sha(PROTOCOL),projection_hash=sha(directory/'projection.json'),opponent_speed=OPPONENT,
        serve_seeds=serve_schedule(),train_wall_seconds=TRAIN_SECONDS,overall_wall_seconds=TOTAL_SECONDS,
        training_camera_cap=24000,evaluation_camera_cap=84000,training_workers=2,evaluation_workers=4,
        calibration_rows_reused=1024,new_calibration_observations=0,long_bank_entries=0,context_fallback_steps=4,
        reserved_final_seeds=list(base.FINAL))
    path=directory/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if any(manifest[k]!=v for k,v in expected.items()):raise ValueError('registered return study differs')
    else:
        manifest=dict(**expected,started_wall_ns=time.time_ns());worker.atomic_json(path,manifest)
    return manifest,learning,evaluation

def execute_training(directory,job,deadline):
    stop=directory/(job['id']+'-registered-stop.json')
    if stop.exists():return json.loads(stop.read_text())
    try:
        while base.complete_report(directory,job) is None:
            remaining=deadline-time.time()
            if remaining<25:return dict(job=job['id'],status='partial')
            raw=rollout.read_report(directory,job['id']);verify=bool(raw and raw['complete'])
            if verify:base.completed_count(directory,job)
            result=rollout.rollout_task(directory,job['id'],0 if verify else 1000,min(160.,remaining-5))
            with (directory/(job['id']+'-costs.jsonl')).open('a') as out:out.write(json.dumps(result)+'\n');out.flush();os.fsync(out.fileno())
            if not result['ok']:raise RuntimeError(result['error'])
            base.pin_completion(directory,job)
        return dict(job=job['id'],status='complete')
    except Exception as error:
        failure=dict(job=job['id'],status='failed',reason=f'{type(error).__name__}: {error}')
        worker.atomic_json(stop,failure);return failure

def execute_evaluation(directory,jobs,deadline):
    pending=[j for j in jobs if base.complete_report(directory,j) is None];active={};failed=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        while pending or active:
            while pending and len(active)<4 and deadline-time.time()>25:
                job=pending.pop(0);report=rollout.read_report(directory,job['id']);verify=bool(report and report['complete'])
                if verify:base.completed_count(directory,job)
                future=pool.submit(rollout.rollout_task,directory,job['id'],0 if verify else 600,min(160.,deadline-time.time()-5))
                active[future]=job
            if not active:break
            done,_=wait(active,timeout=1.,return_when=FIRST_COMPLETED)
            for future in done:
                job=active.pop(future);cost=future.result()
                with (directory/'evaluation-costs.jsonl').open('a') as out:out.write(json.dumps(cost)+'\n');out.flush();os.fsync(out.fileno())
                if not cost['ok']:failed.append(dict(job=job['id'],reason=cost['error']));continue
                base.pin_completion(directory,job)
                if base.complete_report(directory,job) is None:pending.insert(0,job)
    return dict(complete=not pending and not failed,failed=failed,unfinished=[j['id'] for j in pending])

def annotate(directory,jobs):
    result={}
    for job in jobs:
        report=base.complete_report(directory,job)
        if report is None:continue
        with sqlite3.connect(directory/(job['id']+'.db')) as c:rows=[json.loads(r[0]) for r in c.execute('SELECT record FROM replay_samples ORDER BY seq')]
        c=job['config'] if job['kind']=='evaluate' else dict(seed=job['seed'],opponent_speed=OPPONENT,max_episode_ticks=2400)
        rallies=replay_records(rows,seed=c['seed'],opponent_speed=c['opponent_speed'],max_episode_ticks=c['max_episode_ticks'])
        result[job['id']]=dict(rallies=rallies,summary=aggregate(rallies),verified_camera_rows=len(rows))
    return result

def supervise(directory):
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);manifest,learning,evaluation=prepare(directory)
        start=manifest['started_wall_ns']/1e9
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
            training=list(pool.map(execute_training,[directory]*2,learning,[start+TRAIN_SECONDS]*2))
        ready={r['job']:r['status']=='complete' for r in training}
        eligible=[j for j in evaluation if not j['policy'].startswith('hold') or ready.get('learn_'+j['policy'],False)]
        evaluated=execute_evaluation(directory,eligible,start+TOTAL_SECONDS-120.)
        annotations=annotate(directory,[*learning,*eligible]);worker.atomic_json(directory/'return-annotations.json',annotations)
        groups={};by_policy={}
        for policy in dict.fromkeys(j['policy'] for j in evaluation):
            group=[j for j in eligible if j['policy']==policy]
            rallies=[r for j in group if j['id'] in annotations for r in annotations[j['id']]['rallies']]
            by_policy[policy]=rallies
            groups[policy]=dict(complete=len(rallies)==SERVES,**aggregate(rallies))
        counts=dict(learn=sum(base.sample_count(directory,j['id']) for j in learning),evaluate=sum(base.sample_count(directory,j['id']) for j in evaluation))
        assert counts['learn']<=24000 and counts['evaluate']<=84000
        complete=all(r['status']=='complete' for r in training) and evaluated['complete']
        summary=dict(status='complete' if complete else 'partial',counts=counts,training=training,evaluation=evaluated,
            groups=groups,paired_evidence=paired_evidence(by_policy),wall_seconds=time.time()-start,annotations_file='return-annotations.json',
            no_automatic_replication_or_holdout=True,scope='One-seed hit-only return-learning mechanism screen, not standard Pong wins.')
        worker.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare(args.directory)
    else:supervise(args.directory)
