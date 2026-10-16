"""Explicit .25-opponent development screen; original Pong is unchanged."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib,json,os,sqlite3,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from fly_connectome.control.session import source_signature,MODEL_HASH
from fly_connectome.control.durable import runtime_signature
from fly_connectome.control.pong_evaluation import PongEvaluationConfig

TRAIN=(172900127,736187003,1597334677)
DEV=904271833
WORKER=Path(__file__).with_name('run_pong_development.py')


def atomic_json(path,value):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w') as out:
        json.dump(value,out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
    temporary.replace(path)


def declared_jobs():
    jobs=[dict(id=f'learn_{TRAIN[0]}',kind='learn',seed=TRAIN[0],lam=.995,alpha=.001,
               opponent_speed=.25,transitions=6000)]
    for policy in ('initial','up','random','noop'):
        config=PongEvaluationConfig(DEV,opponent_speed=.25,policy_kind=policy,
            action_seed=447201113 if policy=='random' else None,skip_baseline_perception=True)
        jobs.append(dict(id=f'eval_{policy}',kind='evaluate',config=asdict(config),weights_from=None))
    for count in (2000,6000):
        jobs.append(dict(id=f'eval_first_at{count}',kind='evaluate',config=asdict(PongEvaluationConfig(DEV,opponent_speed=.25)),
            weights_from=f'learn_{TRAIN[0]}_at{count}.npy'))
        jobs.append(dict(id=f'eval_blank_at{count}',kind='evaluate',config=asdict(PongEvaluationConfig(DEV,opponent_speed=.25,blank_camera=True)),
            weights_from=f'learn_{TRAIN[0]}_at{count}.npy'))
        for seed in TRAIN[1:]:
            name=f'learn_{seed}_to{count}'
            jobs.append(dict(id=name,kind='learn',seed=seed,lam=.995,alpha=.001,opponent_speed=.25,transitions=count))
            jobs.append(dict(id=f'eval_{seed}_at{count}',kind='evaluate',config=asdict(PongEvaluationConfig(DEV,opponent_speed=.25)),weights_from=f'{name}_at{count}.npy'))
    return jobs


def choose_checkpoint(reports,baselines):
    if len(baselines)!=4 or not all(r['complete'] and r['episodes']==12 for r in baselines):
        raise ValueError('all four complete matched baselines required')
    strongest=max(r['wins'] for r in baselines)
    strongest_hit_wins=max(r['own_hit_followed_wins'] for r in baselines)
    eligible=[(r['wins'],count) for r,count in zip(reports,(2000,6000))
              if r['complete'] and r['episodes']==12 and r['wins']>=strongest+3
              and r['own_hit_followed_wins']>=strongest_hit_wins+3]
    return max(eligible,key=lambda pair:(pair[0],-pair[1])) if eligible else None


def supervise(directory):
    directory.mkdir(parents=True,exist_ok=True);manifest_path=directory/'manifest.json'
    jobs=declared_jobs();worker_hash=hashlib.sha256(WORKER.read_bytes()).hexdigest()
    coordinator_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source=source_signature()
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if (manifest['source']!=source or manifest['runtime']!=runtime_signature() or manifest['jobs']!=jobs
                or manifest['driver_hash']!=worker_hash or manifest['coordinator_hash']!=coordinator_hash):
            raise ValueError('changed declared experiment')
    else:
        manifest=dict(started_wall_ns=time.time_ns(),source=source,runtime=runtime_signature(),model_hash=MODEL_HASH,
            driver_hash=worker_hash,coordinator_hash=coordinator_hash,jobs=jobs,
            protocol='docs/reconstruction/2026-10-15-winnable-pong-screen-protocol.md',
            learning_transition_cap=18000,evaluation_transition_cap=129600,wall_cap_seconds=1800,
            opponent_speed=.25,long_bank_entries=0,context_fallback_steps=4,
            torch_threads=1,chunks_per_worker=5,reserved_final_seeds=[308713321,1203078761,1983461927])
        atomic_json(manifest_path,manifest)
    lookup={job['id']:job for job in jobs}
    def execute(name):
        job=lookup[name];path=directory/(name+'.json')
        while not (path.exists() and json.loads(path.read_text())['complete']):
            remaining=1800-(time.time_ns()-manifest['started_wall_ns'])/1e9
            if remaining<25:return dict(job=name,status='partial',failure='aggregate budget exhausted')
            command=[sys.executable,str(WORKER),'--directory',str(directory),'--worker',name,
                     '--max-work-seconds',str(min(160.,remaining-25)), '--chunks-per-worker','5']
            started=time.monotonic()
            try:
                child=subprocess.run(command,capture_output=True,text=True,timeout=min(180.,remaining-5))
                result=json.loads(child.stdout) if child.returncode==0 else None
                cost=dict(job=name,wall_seconds=time.monotonic()-started,returncode=child.returncode,
                          result=result,stderr=child.stderr[-4000:])
            except subprocess.TimeoutExpired:
                child=None;cost=dict(job=name,wall_seconds=time.monotonic()-started,returncode=None,timeout=True)
            with (directory/(name+'-costs.jsonl')).open('a') as out:
                out.write(json.dumps(cost)+'\n');out.flush();os.fsync(out.fileno())
            print(json.dumps(cost),flush=True)
            if child is None or child.returncode:return dict(job=name,status='failed',failure=cost.get('stderr','timeout'))
        return dict(job=name,status='complete')
    baseline_names=[f'eval_{policy}' for policy in ('initial','up','random','noop')]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(execute,baseline_names))
    decision=None
    if all(row['status']=='complete' for row in results):
        baselines=[json.loads((directory/(name+'.json')).read_text()) for name in baseline_names]
        results.append(execute(f'learn_{TRAIN[0]}'))
        if results[-1]['status']=='complete':
            with ThreadPoolExecutor(max_workers=2) as pool:
                results+=list(pool.map(execute,['eval_first_at2000','eval_first_at6000']))
            if all(row['status']=='complete' for row in results):
                reports=[json.loads((directory/f'eval_first_at{count}.json').read_text()) for count in (2000,6000)]
                selected=choose_checkpoint(reports,baselines)
                if selected is not None:
                    wins,count=selected
                    decision=dict(replicate=True,checkpoint=count,first_seed_wins=wins,
                        strongest_baseline_wins=max(r['wins'] for r in baselines),
                        strongest_baseline_own_hit_wins=max(r['own_hit_followed_wins'] for r in baselines),
                        rule='>=3/12 total and own-hit win lift; most wins, tie earlier')
                    target=directory/'selection.json'
                    if target.exists() and json.loads(target.read_text())!=decision:raise ValueError('changed selected checkpoint')
                    atomic_json(target,decision)
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        results+=list(pool.map(execute,[f'learn_{seed}_to{count}' for seed in TRAIN[1:]]+[f'eval_blank_at{count}']))
                    if all(row['status']=='complete' for row in results):
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            results+=list(pool.map(execute,[f'eval_{seed}_at{count}' for seed in TRAIN[1:]]))
                else:decision=dict(replicate=False,reason='neither checkpoint met paired >=3/12 total and own-hit win lift')
    reports=[];counts=dict(learn=0,evaluate=0)
    for job in jobs:
        path=directory/(job['id']+'.json')
        if path.exists():reports.append(json.loads(path.read_text()))
        db=directory/(job['id']+'.db')
        if db.exists():
            with sqlite3.connect(db) as connection:
                counts[job['kind']]+=connection.execute('SELECT count(*) FROM replay_samples').fetchone()[0]
    assert counts['learn']<=18000 and counts['evaluate']<=129600
    summary=dict(status='complete' if all(row['status']=='complete' for row in results) else 'partial',
        decision=decision,counts=counts,results=results,reports=reports,
        wall_seconds=(time.time_ns()-manifest['started_wall_ns'])/1e9,final_holdout_unused=True)
    atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args();supervise(a.directory)
