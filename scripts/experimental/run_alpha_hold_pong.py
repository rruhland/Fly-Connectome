"""Fixed-camera-budget comparison: lower step size versus held actions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib,json,os,sqlite3,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from fly_connectome.control.session import source_signature,MODEL_HASH
from fly_connectome.control.durable import runtime_signature
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.capping import CAP_VERSION

WORKER=Path(__file__).with_name('run_pong_development.py')
ARMS=('low_alpha','hold4')
def atomic_json(path,value):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w') as out:
        json.dump(value,out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
    temporary.replace(path)

def declared_jobs():
    learning=[];evaluation=[]
    for arm,alpha,hold in (('low_alpha',.0001,1),('hold4',.001,4)):
        learning.append(dict(id=f'learn_{arm}',kind='learn',seed=172900127,alpha=alpha,lam=.995,
            action_hold_frames=hold,opponent_speed=.25,transitions=30000,checkpoints=[6000,15000,30000],
            scaling_from='radial-cap.json',max_abs_q=1000.,compact_reports=True))
        evaluation.append(dict(id=f'eval_{arm}',kind='evaluate',config=asdict(PongEvaluationConfig(904271833,
            episodes=24,opponent_speed=.25,action_hold_frames=hold)),weights_from=f'learn_{arm}_at30000.npy',
            scaling_from='radial-cap.json',compact_reports=True))
    baseline=dict(id='eval_random_hold4',kind='evaluate',config=asdict(PongEvaluationConfig(904271833,
        episodes=24,opponent_speed=.25,policy_kind='random',action_seed=447201113,
        skip_baseline_perception=True,action_hold_frames=4)),weights_from=None,compact_reports=True)
    return learning,[baseline],evaluation

def thresholds(controls):
    if len(controls)!=4 or not all(r['complete'] and r['episodes']==24 for r in controls):
        raise ValueError('four complete matched24-rally controls required')
    return dict(total=max(14,max(r['wins'] for r in controls)+4),
                own_hit=max(6,max(r['own_hit_followed_wins'] for r in controls)+4))

def choose_arm(reports,controls):
    eligible=[];gates={}
    for arm in ARMS:
        report=reports.get(arm)
        if arm not in controls or report is None:continue
        gate=thresholds(controls[arm]);gates[arm]=gate
        if (report['complete'] and report['episodes']==24 and report['wins']>=gate['total']
                and report['own_hit_followed_wins']>=gate['own_hit']):eligible.append((arm,report['wins']))
    selected=max(eligible,key=lambda p:(p[1],p[0]=='low_alpha'))[0] if eligible else None
    return dict(selected_arm=selected,thresholds=gates,qualifying_arms=[p[0] for p in eligible],
        endpoint_cameras=30000,tie_rule='low_alpha',independent_replication_not_allocated=True)

def supervise(directory,reference):
    directory.mkdir(parents=True,exist_ok=True);source=source_signature();runtime=runtime_signature()
    proof=json.loads((directory/'baseline-projection-proof.json').read_text())
    old=json.loads((reference/'manifest.json').read_text())
    if (proof['source']!=source or proof['reference_source']!=old['source']
            or old['runtime']!=runtime or not all(p['exact'] for p in proof['proof'])):
        raise ValueError('source/runtime/baseline replay proof differs')
    capbytes=(directory/'radial-cap.json').read_bytes();cap=json.loads(capbytes)
    if cap['version']!=CAP_VERSION or cap['max_norm2']!=256. or cap['base_transform'] is not None:
        raise ValueError('fixed raw-cap256 required')
    learning,baselines,evaluation=declared_jobs();jobs=learning+baselines+evaluation
    paths=[reference/f'eval_{name}.json' for name in ('initial','up','random','noop')]
    controls=[json.loads(p.read_text()) for p in paths]
    expected=dict(source=source,runtime=runtime,model_hash=MODEL_HASH,jobs=jobs,
        driver_hash=hashlib.sha256(WORKER.read_bytes()).hexdigest(),
        coordinator_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cap_hash=hashlib.sha256(capbytes).hexdigest(),
        baseline_proof_hash=hashlib.sha256((directory/'baseline-projection-proof.json').read_bytes()).hexdigest(),
        reference_report_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    path=directory/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if any(manifest[k]!=v for k,v in expected.items()):raise ValueError('changed declared study')
    else:
        manifest=dict(**expected,started_wall_ns=time.time_ns(),
            protocol='docs/reconstruction/2026-10-16-alpha-hold-pong-protocol.md',
            learning_camera_cap=60000,evaluation_camera_cap=86400,shared_wall_cap_seconds=2100,
            baseline_reference=str(reference),long_bank_entries=0,context_fallback_steps=4,torch_threads=1,
            reserved_final_seeds=[308713321,1203078761,1983461927])
        atomic_json(path,manifest)
    def execute(name):
        report=directory/(name+'.json')
        stopped=directory/(name+'-registered-stop.json')
        if stopped.exists():return json.loads(stopped.read_text())
        while not(report.exists() and json.loads(report.read_text())['complete']):
            remaining=2100-(time.time_ns()-manifest['started_wall_ns'])/1e9
            if remaining<25:return dict(job=name,status='partial',reason='shared wall budget exhausted')
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
            if child is None or child.returncode:
                failure=dict(job=name,status='failed',reason=cost.get('stderr','timeout'))
                atomic_json(stopped,failure)
                return failure
        return dict(job=name,status='complete')
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(execute,[j['id'] for j in learning+baselines]))
    done={r['job']:r['status']=='complete' for r in results}
    evaluable=[j for arm,j in zip(ARMS,evaluation) if done.get(f'learn_{arm}')]
    # A failed arm never prevents the other completed endpoint from evaluation.
    with ThreadPoolExecutor(max_workers=2) as pool:
        results+=list(pool.map(execute,[j['id'] for j in evaluable]))
    report_by_id={j['id']:json.loads((directory/(j['id']+'.json')).read_text()) for j in jobs if (directory/(j['id']+'.json')).exists()}
    groups={'low_alpha':controls}
    held=report_by_id.get('eval_random_hold4')
    if held is not None and held['complete']:
        groups['hold4']=[controls[i] for i in (0,1,3)]+[held]
    decision=choose_arm({arm:report_by_id.get(f'eval_{arm}') for arm in ARMS},groups)
    atomic_json(directory/'selection.json',decision)
    counts=dict(learn=0,evaluate=0)
    for j in jobs:
        db=directory/(j['id']+'.db')
        if db.exists():
            with sqlite3.connect(db) as c:counts[j['kind']]+=c.execute('SELECT count(*) FROM replay_samples').fetchone()[0]
    assert counts['learn']<=60000 and counts['evaluate']<=86400
    summary=dict(status='complete' if len(results)==5 and all(r['status']=='complete' for r in results) else 'partial',
        counts=counts,results=results,reports=report_by_id,decision=decision,controls=groups,
        wall_seconds=(time.time_ns()-manifest['started_wall_ns'])/1e9,final_holdout_unused=True)
    atomic_json(directory/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','counts','decision','wall_seconds')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args();supervise(args.directory,args.reference)
