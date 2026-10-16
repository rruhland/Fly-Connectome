"""Fixed 30k raw-versus-sharing study on explicit .25-opponent Pong."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib,json,os,sqlite3,subprocess,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from fly_connectome.control.session import source_signature,MODEL_HASH
from fly_connectome.control.durable import runtime_signature
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.scaling import validate_scaling

TRAIN=172900127;DEV=904271833
WORKER=Path(__file__).with_name('run_pong_development.py')


def atomic_json(path,value):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w') as out:
        json.dump(value,out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
    temporary.replace(path)


def declared_jobs():
    learning=[];evaluation=[]
    for arm in ('raw','sharing'):
        scale='spatial-transform.json' if arm=='sharing' else None
        learning.append(dict(id=f'learn_{arm}',kind='learn',seed=TRAIN,alpha=.001,lam=.995,
            opponent_speed=.25,transitions=30000,checkpoints=[6000,15000,30000],scaling_from=scale,compact_reports=True))
        evaluation.append(dict(id=f'eval_{arm}',kind='evaluate',config=asdict(PongEvaluationConfig(DEV,episodes=24,opponent_speed=.25)),
            weights_from=f'learn_{arm}_at30000.npy',scaling_from=scale,compact_reports=True))
    baselines=[]
    for policy in ('initial','up','random','noop'):
        config=PongEvaluationConfig(DEV,episodes=24,opponent_speed=.25,policy_kind=policy,
            action_seed=447201113 if policy=='random' else None,skip_baseline_perception=True)
        baselines.append(dict(id=f'eval_{policy}',kind='evaluate',config=asdict(config),weights_from=None,compact_reports=True))
    return learning,baselines,evaluation


def choose_arm(reports,baselines):
    if len(baselines)!=4 or not all(r['complete'] and r['episodes']==24 for r in baselines):
        raise ValueError('all four complete matched24-rally baselines required')
    strongest=max(r['wins'] for r in baselines)
    strongest_hits=max(r['own_hit_followed_wins'] for r in baselines)
    eligible=[(arm,r['wins']) for arm,r in zip(('raw','sharing'),reports)
        if r['complete'] and r['episodes']==24 and r['wins']>=strongest+4
        and r['own_hit_followed_wins']>=strongest_hits+4]
    return max(eligible,key=lambda pair:(pair[1],pair[0]=='raw'))[0] if eligible else None


def verify_prefix(old,new,count):
    with sqlite3.connect(old) as previous,sqlite3.connect(new) as current:
        left=previous.execute('SELECT record FROM replay_samples ORDER BY seq LIMIT ?',(count,)).fetchall()
        right=current.execute('SELECT record FROM replay_samples ORDER BY seq LIMIT ?',(count,)).fetchall()
    if len(left)!=count or left!=right:raise ValueError('reference transition prefix differs')


def supervise(directory,reference,calibration):
    directory.mkdir(parents=True,exist_ok=True)
    learning,baselines,evaluation=declared_jobs();jobs=learning+baselines+evaluation
    source=source_signature();runtime=runtime_signature()
    old=json.loads((reference/'manifest.json').read_text())
    if old['source']!=source or old['runtime']!=runtime or old['model_hash']!=MODEL_HASH:
        raise ValueError('reference control source/runtime/model differs')
    fitted=(calibration/'spatial-transform.json').read_bytes();validate_scaling(json.loads(fitted))
    target=directory/'spatial-transform.json'
    if target.exists() and target.read_bytes()!=fitted:raise ValueError('changed calibration artifact')
    if not target.exists():target.write_bytes(fitted)
    expected=dict(source=source,runtime=runtime,model_hash=MODEL_HASH,
        driver_hash=hashlib.sha256(WORKER.read_bytes()).hexdigest(),
        coordinator_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),jobs=jobs,
        calibration_hash=hashlib.sha256(fitted).hexdigest())
    path=directory/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if any(manifest[k]!=v for k,v in expected.items()):raise ValueError('changed declared experiment')
    else:
        manifest=dict(**expected,started_wall_ns=time.time_ns(),
            protocol='docs/reconstruction/2026-10-15-pong-exposure-sharing-protocol.md',
            learning_transition_cap=60000,evaluation_transition_cap=172800,wall_cap_seconds=2700,
            opponent_speed=.25,long_bank_entries=0,context_fallback_steps=4,torch_threads=1,
            reference_directory=str(reference),calibration_directory=str(calibration),
            calibrated_transitions_accounted_separately=1024,reserved_final_seeds=[308713321,1203078761,1983461927])
        atomic_json(path,manifest)
    lookup={j['id']:j for j in jobs}
    def execute(name):
        report=directory/(name+'.json')
        while not(report.exists() and json.loads(report.read_text())['complete']):
            remaining=2700-(time.time_ns()-manifest['started_wall_ns'])/1e9
            if remaining<25:return dict(job=name,status='partial',failure='aggregate budget exhausted')
            started=time.monotonic();child=None
            try:
                child=subprocess.run([sys.executable,str(WORKER),'--directory',str(directory),'--worker',name,
                    '--max-work-seconds',str(min(160.,remaining-25)),'--chunks-per-worker','5'],
                    capture_output=True,text=True,timeout=min(180.,remaining-5))
                cost=dict(job=name,wall_seconds=time.monotonic()-started,returncode=child.returncode,
                    result=json.loads(child.stdout) if child.returncode==0 else None,stderr=child.stderr[-4000:])
            except subprocess.TimeoutExpired:
                cost=dict(job=name,wall_seconds=time.monotonic()-started,returncode=None,timeout=True)
            with (directory/(name+'-costs.jsonl')).open('a') as out:
                out.write(json.dumps(cost)+'\n');out.flush();os.fsync(out.fileno())
            print(json.dumps(cost),flush=True)
            if child is None or child.returncode:return dict(job=name,status='failed',failure=cost.get('stderr','timeout'))
        return dict(job=name,status='complete')
    def simple_controls():return [execute(j['id']) for j in baselines]
    with ThreadPoolExecutor(max_workers=3) as pool:
        raw=pool.submit(execute,'learn_raw');sharing=pool.submit(execute,'learn_sharing')
        simple=pool.submit(simple_controls)
        results=[raw.result(),sharing.result()]+simple.result()
    prefix_checks={};decision=None
    if all(r['status']=='complete' for r in results):
        verify_prefix(reference/f'learn_{TRAIN}.db',directory/'learn_raw.db',6000)
        if not np.array_equal(np.load(reference/f'learn_{TRAIN}_at6000.npy',allow_pickle=False),
                              np.load(directory/'learn_raw_at6000.npy',allow_pickle=False)):
            raise ValueError('raw6k weights differ from reference')
        prefix_checks['raw6000_records_and_weights_exact']=True
        for job in baselines:
            with sqlite3.connect(reference/(job['id']+'.db')) as connection:
                count=connection.execute('SELECT count(*) FROM replay_samples').fetchone()[0]
            verify_prefix(reference/(job['id']+'.db'),directory/(job['id']+'.db'),count)
        prefix_checks['all_simple_control_first12_records_exact']=True
        atomic_json(directory/'prefix-checks.json',prefix_checks)
        with ThreadPoolExecutor(max_workers=2) as pool:results+=list(pool.map(execute,[j['id'] for j in evaluation]))
        if all(r['status']=='complete' for r in results):
            reports=[json.loads((directory/(j['id']+'.json')).read_text()) for j in evaluation]
            controls=[json.loads((directory/(j['id']+'.json')).read_text()) for j in baselines]
            selected=choose_arm(reports,controls)
            decision=dict(selected_arm=selected,checkpoint=30000,
                reason='>=4/24 total and own-hit win lift; most wins, tie raw' if selected else 'neither arm met both baseline lift screens',
                independent_seed_replication_not_yet_allocated=True)
            atomic_json(directory/'selection.json',decision)
    reports=[];counts=dict(learn=0,evaluate=0)
    for job in jobs:
        report=directory/(job['id']+'.json');db=directory/(job['id']+'.db')
        if report.exists():reports.append(json.loads(report.read_text()))
        if db.exists():
            with sqlite3.connect(db) as connection:counts[job['kind']]+=connection.execute('SELECT count(*) FROM replay_samples').fetchone()[0]
    assert counts['learn']<=60000 and counts['evaluate']<=172800
    summary=dict(status='complete' if len(results)==len(jobs) and all(r['status']=='complete' for r in results) else 'partial',
        counts=counts,results=results,reports=reports,decision=decision,prefix_checks=prefix_checks,
        wall_seconds=(time.time_ns()-manifest['started_wall_ns'])/1e9,final_holdout_unused=True)
    atomic_json(directory/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','counts','decision','wall_seconds')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    args=parser.parse_args();supervise(args.directory,args.reference,args.calibration)
