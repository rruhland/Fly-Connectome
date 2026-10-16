"""Execute the predeclared score-free block-conditioning toy mechanism gate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib,json,os,random,sqlite3,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from fly_connectome.control.durable import DurableRun,runtime_signature
from fly_connectome.control.session import source_signature,MODEL_HASH
from fly_connectome.control.toy_session import ToySpec,ToySession
from fly_connectome.control.scaling import SCALING_VERSION,energy_matched_scales

TRAIN=(531028119,1047199272,1958971091)
EVAL=(62697349,1197177281,2007311671)
RANDOM=(704719329,1310920477,2031518761)


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as out:
        json.dump(value,out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
    temporary.replace(path)


def jobs(episodes=25,*,include_calibration=True,include_baselines=True):
    result=[]
    if include_calibration:
        result.append(dict(id='calibration',stage='calibration',conditioned=False,weights_from=None,
            spec=asdict(ToySpec('delayed',482977133,64,mode='evaluate',policy_kind='random',action_seed=143332057,diagnostic=True))))
    for train,evaluate,random_seed in zip(TRAIN,EVAL,RANDOM):
        for conditioned in (False,True):
            label=f'{train}_'+('conditioned' if conditioned else 'raw')
            result.append(dict(id='learn_'+label,stage='learn',conditioned=conditioned,weights_from=None,
                spec=asdict(ToySpec('delayed',train,episodes,diagnostic=True))))
            result.append(dict(id='eval_'+label,stage='evaluate',conditioned=conditioned,weights_from='learn_'+label,
                spec=asdict(ToySpec('delayed',evaluate,20,mode='evaluate',diagnostic=True))))
        for kind in (('initial','random') if include_baselines else ()):
            result.append(dict(id=f'eval_{evaluate}_{kind}',stage='evaluate',conditioned=False,weights_from=None,
                spec=asdict(ToySpec('delayed',evaluate,20,mode='evaluate',policy_kind=kind,
                    action_seed=random_seed if kind=='random' else None,diagnostic=True))))
    expected=6*episodes*8+960+(512 if include_calibration else 0)+(960 if include_baselines else 0)
    assert sum(ToySpec(**row['spec']).transition_budget for row in result)==expected
    return result


def worker(directory,job_id,max_seconds=160.):
    if not 0<max_seconds<=160:raise ValueError('bounded work budget required')
    torch.set_num_threads(1);started=time.monotonic()
    manifest=json.loads((directory/'manifest.json').read_text())
    if (manifest['source']!=source_signature() or manifest['runtime']!=runtime_signature()
            or manifest['driver_hash']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError('changed source/runtime/driver')
    job=next(j for j in manifest['jobs'] if j['id']==job_id)
    settings=dict(job['spec'])
    if job['conditioned']:settings['policy_scaling']=json.loads((directory/'scaling.json').read_text())
    spec=ToySpec(**settings);spec.validate();path=directory/(job_id+'.db')
    if path.exists():run=DurableRun(path)
    else:
        random.seed(spec.seed);np.random.seed(spec.seed);torch.manual_seed(spec.seed)
        weights=None if job['weights_from'] is None else np.load(directory/(job['weights_from']+'.npy'),allow_pickle=False)
        if weights is not None:
            training=json.loads((directory/(job['weights_from']+'.json')).read_text())
            if hashlib.sha256(weights.tobytes()).hexdigest()!=training['weights_hash']:
                raise ValueError('frozen evaluation weights differ from completed training')
        run=DurableRun(path,session=ToySession(spec,weights=weights))
    session=run.session
    if session.spec!=spec:raise ValueError('journal configuration differs')
    before=session.transitions;replayed=run.replayed
    while not session.complete and session.transitions-before<200 and time.monotonic()-started<max_seconds:run.step()
    rows=[json.loads(r[0]) for r in run.db.execute('SELECT record FROM replay_samples ORDER BY seq')]
    scoring=[r for r in rows if r['terminated']]
    report=dict(job=job_id,stage=job['stage'],spec=asdict(spec),conditioned=job['conditioned'],
        complete=session.complete,transitions=session.transitions,new_transitions=session.transitions-before,
        replayed_transitions=replayed,episodes=session.episodes,initial_cameras=session.initial_cameras,
        transition_cameras=session.transition_cameras,total_reward=sum(r['reward'] for r in rows),
        side_results={str(side):dict(correct=sum(r['reward']>0 for r in scoring if r['target_side']==side),
            episodes=sum(r['target_side']==side for r in scoring)) for side in (0,1)})
    if job['stage']=='learn' and session.complete:
        target=directory/(job_id+'.npy');temporary=target.with_suffix('.tmp')
        with temporary.open('wb') as out:np.save(out,session.learner.weights,allow_pickle=False);out.flush();os.fsync(out.fileno())
        temporary.replace(target)
        report['weight_norm']=float(np.linalg.norm(session.learner.weights))
        report['weights_hash']=hashlib.sha256(session.learner.weights.tobytes()).hexdigest()
    run.stop();report['worker_wall_seconds']=time.monotonic()-started
    atomic_json(directory/(job_id+'.json'),report);print(json.dumps(report),flush=True)


def fit(directory):
    with sqlite3.connect(directory/'calibration.db') as db:rows=[json.loads(r[0]) for r in db.execute('SELECT record FROM replay_samples ORDER BY seq')]
    if len(rows)!=512:raise ValueError('incomplete declared calibration')
    names=tuple(rows[0]['input_block_norm2'])
    means={name:sum(row['input_block_norm2'][name] for row in rows)/len(rows) for name in names}
    scales=energy_matched_scales(means)
    config=dict(version=SCALING_VERSION,scales=scales,mean_squared_norms=means,minimum=.125,maximum=8.,solver_iterations=80,
        energy_before=sum(means.values()),energy_after=sum(means[name]*scales[name]**2 for name in names),
        calibration=dict(task='delayed',seed=482977133,action_seed=143332057,episodes=64,transitions=512,
            statistic='pre-action raw block squared norm; rewards ignored',model_hash=MODEL_HASH,
            source_digest=hashlib.sha256(json.dumps(source_signature(),sort_keys=True).encode()).hexdigest()))
    path=directory/'scaling.json'
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('fitted scaling differs')
    atomic_json(path,config)


def supervise(directory,*,exposure=False,reuse_from=None):
    directory.mkdir(parents=True,exist_ok=True);path=directory/'manifest.json'
    expected=jobs(200,include_calibration=False,include_baselines=False) if exposure else jobs()
    driver_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cap=10560 if exposure else 3632
    if exposure:
        if reuse_from is None:raise ValueError('exposure diagnostic needs the completed mechanism artifact directory')
        origin=json.loads((reuse_from/'manifest.json').read_text())
        previous=json.loads((reuse_from/'summary.json').read_text())
        if (origin['source']!=source_signature() or origin['runtime']!=runtime_signature()
                or origin['model_hash']!=MODEL_HASH or previous['status']!='complete'):
            raise ValueError('reuse requires identical source and complete original mechanism allocation')
        scaling=json.loads((reuse_from/'scaling.json').read_text())
        if (directory/'scaling.json').exists() and json.loads((directory/'scaling.json').read_text())!=scaling:
            raise ValueError('reused calibration differs')
        atomic_json(directory/'scaling.json',scaling)
    if path.exists():
        manifest=json.loads(path.read_text())
        if (manifest['source']!=source_signature() or manifest['runtime']!=runtime_signature()
                or manifest['driver_hash']!=driver_hash or manifest['jobs']!=expected):raise ValueError('changed experiment')
    else:
        manifest=dict(started_wall_ns=time.time_ns(),source=source_signature(),runtime=runtime_signature(),
            driver_hash=driver_hash,model_hash=MODEL_HASH,jobs=expected,transition_cap=cap,wall_cap_seconds=1200,
            protocol='docs/reconstruction/2026-10-15-fixed-block-conditioning-protocol.md',
            long_bank_entries=0,context_fallback_steps=4,torch_threads=1)
        if exposure:
            manifest.update(protocol='docs/reconstruction/2026-10-15-conditioning-exposure-protocol.md',
                reused_calibration_and_baselines=str(reuse_from),
                reused_scaling_sha256=hashlib.sha256((reuse_from/'scaling.json').read_bytes()).hexdigest())
        atomic_json(path,manifest)
    def execute(job):
        report_path=directory/(job['id']+'.json')
        while not (report_path.exists() and json.loads(report_path.read_text())['complete']):
            remaining=1200-(time.time_ns()-manifest['started_wall_ns'])/1e9
            if remaining<25:return dict(job=job['id'],status='partial',failure='aggregate budget exhausted')
            command=[sys.executable,__file__,'--directory',str(directory),'--worker',job['id'],'--max-seconds',str(min(160.,remaining-25))]
            started=time.monotonic()
            try:
                result=subprocess.run(command,capture_output=True,text=True,timeout=min(180.,remaining-5))
                child=json.loads(result.stdout) if result.returncode==0 else {}
                cost=dict(returncode=result.returncode,stderr=result.stderr[-4000:],
                    wall_seconds=time.monotonic()-started,new_transitions=child.get('new_transitions'),
                    replayed_transitions=child.get('replayed_transitions'))
            except subprocess.TimeoutExpired:
                result=None;cost=dict(returncode=None,timeout=True,wall_seconds=time.monotonic()-started)
            with (directory/(job['id']+'-costs.jsonl')).open('a') as out:
                out.write(json.dumps(cost)+'\n');out.flush();os.fsync(out.fileno())
            if result is None:return dict(job=job['id'],status='partial',failure='worker timed out')
            if result.returncode:return dict(job=job['id'],status='failed',failure=result.stderr[-4000:])
        return dict(job=job['id'],status='complete')
    results=[]
    for stage in ('calibration','learn','evaluate'):
        stage_jobs=[job for job in expected if job['stage']==stage]
        with ThreadPoolExecutor(max_workers=3) as pool:completed=list(pool.map(execute,stage_jobs))
        results+=completed
        if not all(row['status']=='complete' for row in completed):break
        if stage=='calibration' and stage_jobs:fit(directory)
    reports=[json.loads((directory/(job['id']+'.json')).read_text()) for job in expected if (directory/(job['id']+'.json')).exists()]
    counts={stage:0 for stage in ('calibration','learn','evaluate')}
    for job in expected:
        db=directory/(job['id']+'.db')
        if db.exists():
            with sqlite3.connect(db) as connection:
                counts[job['stage']]+=connection.execute('SELECT count(*) FROM replay_samples').fetchone()[0]
    assert sum(counts.values())<=cap
    summary=dict(status='complete' if len(results)==len(expected) and all(r['status']=='complete' for r in results) else 'partial',
        counts=counts,results=results,reports=reports,wall_seconds=(time.time_ns()-manifest['started_wall_ns'])/1e9)
    atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--worker');p.add_argument('--max-seconds',type=float,default=160.)
    p.add_argument('--exposure',action='store_true');p.add_argument('--reuse-from',type=Path);a=p.parse_args()
    if a.worker:worker(a.directory,a.worker,a.max_seconds)
    else:supervise(a.directory,exposure=a.exposure,reuse_from=a.reuse_from)
