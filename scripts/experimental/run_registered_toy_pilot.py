"""UNVALIDATED supervisor scaffold: manifest-tested, never executed end to end.

Do not launch automatically. Ryan paused experiments to replace orchestration.
"""
import argparse
from dataclasses import asdict
import hashlib,json,os,random,sqlite3,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from fly_connectome.control.durable import DurableRun,runtime_signature
from fly_connectome.control.toy_session import ToySpec,ToySession
from fly_connectome.control.session import source_signature,MODEL_HASH
from fly_connectome.control.codec import fingerprint


def jobs():
    rows=[]
    for task,base in [('immediate',2101),('delayed',2201)]:
        for seed in range(base,base+3):
            for lam in (.8,0.):
                label=f'{task}_{seed}_lambda{lam}'
                rows.append(dict(id='learn_'+label,spec=asdict(ToySpec(task,seed,200 if task=='immediate' else 25,lam=lam)),weights_from=None))
                rows.append(dict(id='eval_'+label,spec=asdict(ToySpec(task,seed+1000,20,mode='evaluate',lam=lam)),weights_from='learn_'+label))
        for seed in range(base+1000,base+1003):
            for policy in ('initial','random'):
                rows.append(dict(id=f'eval_{task}_{seed}_{policy}',
                                 spec=asdict(ToySpec(task,seed,20,mode='evaluate',policy_kind=policy,
                                                   action_seed=seed+1000 if policy=='random' else None)),weights_from=None))
    assert sum(ToySpec(**r['spec']).transition_budget for r in rows)==4560
    assert sum(ToySpec(**r['spec']).transition_budget for r in rows if r['spec']['mode']=='learn')==2400
    return rows


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as out:
        json.dump(value,out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
    temporary.replace(path)


def worker(directory,job_id,max_work_seconds=160.):
    started=time.monotonic();torch.set_num_threads(1)
    manifest=json.loads((directory/'manifest.json').read_text())
    job=next(x for x in manifest['jobs'] if x['id']==job_id)
    spec=ToySpec(**job['spec']);spec.validate()
    db=directory/(job_id+'.db')
    if db.exists():run=DurableRun(db)
    else:
        random.seed(spec.seed);np.random.seed(spec.seed);torch.manual_seed(spec.seed)
        weights=None
        if job['weights_from']:weights=np.load(directory/(job['weights_from']+'.npy'),allow_pickle=False)
        run=DurableRun(db,session=ToySession(spec,weights=weights))
    if run.session.spec!=spec:raise ValueError('journal differs from registered job')
    before=run.session.transitions;replayed=run.replayed
    # Reserve 20 seconds for persistence before the supervisor's 180-second timeout.
    while not run.session.complete and run.session.transitions-before<200 and time.monotonic()-started<max_work_seconds:
        run.step()
    s=run.session
    rows=[json.loads(x[0]) for x in run.db.execute('SELECT record FROM replay_samples ORDER BY seq')]
    scoring=[x for x in rows if x['terminated']]
    report=dict(job=job_id,spec=asdict(spec),complete=s.complete,transitions=s.transitions,
                new_transitions=s.transitions-before,replayed_transitions=replayed,
                episodes=s.episodes,initial_cameras=s.initial_cameras,transition_cameras=s.transition_cameras,
                total_reward=sum(x['reward'] for x in rows),
                side_results={str(side):dict(correct=sum(x['reward']>0 for x in scoring if x['target_side']==side),
                                              episodes=sum(x['target_side']==side for x in scoring)) for side in (0,1)},
                state_hash=fingerprint(s),timing_prefixes=run.timing_summary())
    if spec.mode=='learn' and s.complete:
        np.save(directory/(job_id+'.npy'),s.learner.weights,allow_pickle=False)
        report['weights_hash']=hashlib.sha256(s.learner.weights.tobytes()).hexdigest()
    run.stop()
    report['worker_wall_seconds']=time.monotonic()-started
    atomic_json(directory/(job_id+'.json'),report)
    print(json.dumps(report),flush=True)


def supervise(directory):
    directory.mkdir(parents=True,exist_ok=True)
    manifest_path=directory/'manifest.json';expected=jobs()
    driver_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['jobs']!=expected or manifest['source']!=source_signature() or manifest['runtime']!=runtime_signature() or manifest.get('driver_hash')!=driver_hash:
            raise ValueError('changed registration/source/runtime')
    else:
        manifest=dict(started_wall_ns=time.time_ns(),source=source_signature(),runtime=runtime_signature(),
                      model_hash=MODEL_HASH,driver_hash=driver_hash,jobs=expected,aggregate_seconds=3600,new_transition_cap=4560)
        atomic_json(manifest_path,manifest)
    costs_path=directory/'process-costs.jsonl';failure=None
    for job in expected:
        report_path=directory/(job['id']+'.json')
        while True:
            if report_path.exists() and json.loads(report_path.read_text())['complete']:break
            remaining=3600-(time.time_ns()-manifest['started_wall_ns'])/1e9
            if remaining<25:
                failure='aggregate wall budget exhausted; partial allocations retained';break
            started=time.monotonic()
            command=[sys.executable,__file__,'--worker',job['id'],'--directory',str(directory),
                     '--max-work-seconds',str(min(160.,remaining-25.))]
            try:
                result=subprocess.run(command,capture_output=True,text=True,timeout=min(180,remaining-5))
                cost=dict(job=job['id'],wall_seconds=time.monotonic()-started,returncode=result.returncode,
                          timeout=False,stderr=result.stderr[-4000:])
            except subprocess.TimeoutExpired:
                cost=dict(job=job['id'],wall_seconds=time.monotonic()-started,returncode=None,timeout=True);result=None
            with costs_path.open('a') as out:
                out.write(json.dumps(cost)+'\n');out.flush();os.fsync(out.fileno())
            print(json.dumps(dict(progress=job['id'],**{k:v for k,v in cost.items() if k!='job'})),flush=True)
            if result is not None and result.returncode:
                failure='worker failed; no automatic extension/tuning';break
            if result is None:continue
        if failure:break
    reports=[]
    for job in expected:
        path=directory/(job['id']+'.json')
        if path.exists():reports.append(json.loads(path.read_text()))
    comparisons=[]
    for seed in (2101,2102,2103):
        names=[f'learn_immediate_{seed}_lambda{lam}' for lam in (.8,0.)]
        if all((directory/(name+'.npy')).exists() for name in names):
            with sqlite3.connect(directory/(names[0]+'.db')) as a,sqlite3.connect(directory/(names[1]+'.db')) as b:
                identical=a.execute('SELECT record FROM replay_samples ORDER BY seq').fetchall()==b.execute('SELECT record FROM replay_samples ORDER BY seq').fetchall()
            equal=np.array_equal(np.load(directory/(names[0]+'.npy')),np.load(directory/(names[1]+'.npy')))
            comparisons.append(dict(seed=seed,identical_trajectories=identical,identical_weights=equal))
            if not identical or not equal:failure='immediate-arm invariant failed'
    actual_counts={}
    for job in expected:
        db=directory/(job['id']+'.db')
        if db.exists():
            with sqlite3.connect(db) as connection:
                actual_counts[job['id']]=connection.execute('SELECT count(*) FROM replay_samples').fetchone()[0]
    total=sum(actual_counts.values())
    if total>4560:raise AssertionError('allocation cap exceeded')
    summary=dict(status='partial' if failure or len(reports)!=len(expected) or not all(r['complete'] for r in reports) else 'complete',
                 failure=failure,total_committed_new_transitions=total,committed_counts=actual_counts,
                 wall_seconds=(time.time_ns()-manifest['started_wall_ns'])/1e9,reports=reports,
                 immediate_arm_comparisons=comparisons)
    atomic_json(directory/'summary.json',summary)
    print(json.dumps(dict(final_status=summary['status'],new_transitions=total,wall_seconds=summary['wall_seconds'],failure=failure)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',type=Path)
    p.add_argument('--directory',type=Path);p.add_argument('--worker');p.add_argument('--max-work-seconds',type=float,default=160.);args=p.parse_args()
    if args.pilot:supervise(args.pilot)
    elif args.worker and args.directory:worker(args.directory,args.worker,args.max_work_seconds)
    else:p.error('provide --pilot or --worker/--directory')
