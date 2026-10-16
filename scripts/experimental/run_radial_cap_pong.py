"""Single fixed30k raw+radial-cap experiment; matched controls are reused."""
import argparse
from dataclasses import asdict
import hashlib,json,os,sqlite3,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from fly_connectome.control.session import source_signature,MODEL_HASH
from fly_connectome.control.durable import runtime_signature
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.capping import CAP_VERSION

WORKER=Path(__file__).with_name('run_pong_development.py')


def atomic_json(path,value):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w') as out:
        json.dump(value,out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
    temporary.replace(path)


def declared_jobs():
    return [dict(id='learn_raw_cap',kind='learn',seed=172900127,alpha=.001,lam=.995,opponent_speed=.25,
                transitions=30000,checkpoints=[6000,15000,30000],compact_reports=True,
                scaling_from='radial-cap.json',max_abs_q=1000.),
            dict(id='eval_raw_cap',kind='evaluate',config=asdict(PongEvaluationConfig(904271833,episodes=24,opponent_speed=.25)),
                weights_from='learn_raw_cap_at30000.npy',scaling_from='radial-cap.json',compact_reports=True)]


def promising(report,controls):
    if len(controls)!=4 or not all(c['complete'] and c['episodes']==24 for c in controls):
        raise ValueError('four complete matched24-rally controls required')
    return (report['complete'] and report['episodes']==24
        and report['wins']>=max(c['wins'] for c in controls)+4
        and report['own_hit_followed_wins']>=max(c['own_hit_followed_wins'] for c in controls)+4)


def supervise(directory,reference):
    directory.mkdir(parents=True,exist_ok=True);source=source_signature();runtime=runtime_signature()
    proof=json.loads((directory/'cross-source-baseline-proof.json').read_text())
    if proof['source']!=source or not all(p['all_records_bitexact'] for p in proof['proof'].values()):
        raise ValueError('verified baseline source/action/physics proof required')
    old=json.loads((reference/'manifest.json').read_text())
    if old['source']!=proof['reference_source'] or old['runtime']!=runtime:
        raise ValueError('baseline reference/runtime differs')
    cap=(directory/'radial-cap.json').read_bytes();config=json.loads(cap)
    if config['version']!=CAP_VERSION or config['max_norm2']!=256. or config['base_transform'] is not None:
        raise ValueError('exact fixed raw+cap256 required')
    control_paths=[reference/f'eval_{policy}.json' for policy in ('initial','up','random','noop')]
    controls=[json.loads(path.read_text()) for path in control_paths]
    expected=dict(source=source,runtime=runtime,model_hash=MODEL_HASH,jobs=declared_jobs(),
        driver_hash=hashlib.sha256(WORKER.read_bytes()).hexdigest(),
        coordinator_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cap_hash=hashlib.sha256(cap).hexdigest(),
        baseline_proof_hash=hashlib.sha256((directory/'cross-source-baseline-proof.json').read_bytes()).hexdigest(),
        baseline_report_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in control_paths})
    path=directory/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if any(manifest[k]!=v for k,v in expected.items()):raise ValueError('changed declared study')
    else:
        manifest=dict(**expected,started_wall_ns=time.time_ns(),
            protocol='docs/reconstruction/2026-10-16-radial-cap-pong-protocol.md',
            learning_transition_cap=30000,evaluation_transition_cap=28800,wall_cap_seconds=2100,
            baseline_reference=str(reference),long_bank_entries=0,context_fallback_steps=4,
            torch_threads=1,calibration_transitions_reused=1024,reserved_final_seeds=[308713321,1203078761,1983461927])
        atomic_json(path,manifest)
    results=[]
    for job in manifest['jobs']:
        report=directory/(job['id']+'.json')
        while not(report.exists() and json.loads(report.read_text())['complete']):
            remaining=2100-(time.time_ns()-manifest['started_wall_ns'])/1e9
            if remaining<25:
                results.append(dict(job=job['id'],status='partial',reason='aggregate wall budget exhausted'));break
            started=time.monotonic();child=None
            try:
                child=subprocess.run([sys.executable,str(WORKER),'--directory',str(directory),'--worker',job['id'],
                    '--max-work-seconds',str(min(160.,remaining-25)),'--chunks-per-worker','5'],
                    capture_output=True,text=True,timeout=min(180.,remaining-5))
                cost=dict(job=job['id'],wall_seconds=time.monotonic()-started,returncode=child.returncode,
                    result=json.loads(child.stdout) if child.returncode==0 else None,stderr=child.stderr[-4000:])
            except subprocess.TimeoutExpired:
                cost=dict(job=job['id'],wall_seconds=time.monotonic()-started,returncode=None,timeout=True)
            with (directory/(job['id']+'-costs.jsonl')).open('a') as out:
                out.write(json.dumps(cost)+'\n');out.flush();os.fsync(out.fileno())
            print(json.dumps(cost),flush=True)
            if child is None or child.returncode:
                results.append(dict(job=job['id'],status='failed',reason=cost.get('stderr','timeout')));break
        else:results.append(dict(job=job['id'],status='complete'))
        if results[-1]['status']!='complete':break
    reports=[];counts=dict(learn=0,evaluate=0)
    for job in manifest['jobs']:
        report=directory/(job['id']+'.json');db=directory/(job['id']+'.db')
        if report.exists():reports.append(json.loads(report.read_text()))
        if db.exists():
            with sqlite3.connect(db) as connection:counts[job['kind']]+=connection.execute('SELECT count(*) FROM replay_samples').fetchone()[0]
    assert counts['learn']<=30000 and counts['evaluate']<=28800
    complete=len(results)==2 and all(r['status']=='complete' for r in results)
    decision=dict(promising=promising(reports[-1],controls),endpoint=30000,independent_replication_not_allocated=True) if complete else None
    if decision is not None:atomic_json(directory/'selection.json',decision)
    summary=dict(status='complete' if complete else 'partial',counts=counts,reports=reports,results=results,
        controls=controls,decision=decision,wall_seconds=(time.time_ns()-manifest['started_wall_ns'])/1e9,final_holdout_unused=True)
    atomic_json(directory/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','counts','decision','wall_seconds')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args();supervise(args.directory,args.reference)
