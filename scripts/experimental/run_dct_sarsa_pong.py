"""One frozen DCT/SARSA condition, three prespecified 30k-camera training seeds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import fcntl,hashlib,json,os,sqlite3,subprocess,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from fly_connectome.control.session import source_signature,MODEL_HASH
from fly_connectome.control.durable import runtime_signature
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.projection import FixedPolicyProjection
from run_pong_development import atomic_json
from run_pong_policy_search_continuation import journal_hash

ROOT=Path(__file__).resolve().parents[2]
WORKER=Path(__file__).with_name('run_pong_development.py')
PROTOCOL=ROOT/'docs/reconstruction/2026-10-16-dct-sarsa-pong-protocol.md'
REFERENCE=Path('/workspace/m1b-experiments/pong-exposure-sharing-20261015-r1')
TRANSFORM=Path('/workspace/m1b-experiments/pong-score-policy-search-20261016-r1/search-transform.npy')
FIT=TRANSFORM.with_name('transform-fit.json')
PREFLIGHT=Path('/workspace/m1b-experiments/dct-sarsa-preflight-20261016/report.json')
INTEGRITY_HELPER=Path(__file__).with_name('run_pong_policy_search_continuation.py')
FINAL=(308713321,1203078761,1983461927)
TRAINING_SEEDS=(172900127,736187003,1597334677)
TRAINING_SECONDS=1800.;TOTAL_SECONDS=2100.
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def declared_jobs():
    learning=[dict(id=f'learn_dct_{seed}',kind='learn',seed=seed,alpha=.001,lam=.995,
                 opponent_speed=.25,transitions=30000,checkpoints=[30000],compact_reports=True,
                 projection_from='projection.json',max_abs_q=1000.) for seed in TRAINING_SEEDS]
    evaluation=[dict(id=f'eval_dct_{seed}',kind='evaluate',config=asdict(PongEvaluationConfig(904271833,
                 episodes=24,opponent_speed=.25)),weights_from=f'learn_dct_{seed}_at30000.npy',
                 scaling_from=f'learn_dct_{seed}_at30000.cap.json',compact_reports=True) for seed in TRAINING_SEEDS]
    return learning+evaluation

def thresholds(controls):
    if len(controls)!=4 or not all(c['complete'] and c['episodes']==24 for c in controls):
        raise ValueError('four complete matched24-rally controls required')
    return dict(total=max(14,max(c['wins'] for c in controls)+4),
                own_hit=max(6,max(c['own_hit_followed_wins'] for c in controls)+4))

def condition_decision(reports,controls):
    gate=thresholds(controls)
    outcomes={str(seed):bool((r:=reports.get(f'eval_dct_{seed}')) and r['complete'] and r['episodes']==24
        and r['wins']>=gate['total'] and r['own_hit_followed_wins']>=gate['own_hit']) for seed in TRAINING_SEEDS}
    return dict(gate=gate,per_seed_pass=outcomes,qualifies=all(outcomes.values()),
        rule='all three prespecified endpoints pass; no best-seed selection or replacement')

def sample_count(directory,name):
    path=directory/(name+'.db')
    if not path.exists():return 0
    with sqlite3.connect(path) as c:return c.execute('SELECT count(*) FROM replay_samples').fetchone()[0]

def endpoint_artifacts(directory,job):
    if job['kind']=='learn':
        stem=f"{job['id']}_at{job['transitions']}"
        paths=[directory/(stem+suffix) for suffix in ('.npy','.json','.theta.npy','.cap.json')]
        projection=FixedPolicyProjection(4389,json.loads((directory/'projection.json').read_text()))
        theta=np.load(paths[2],allow_pickle=False)
        if theta.shape!=(3,projection.output_dimension):raise ValueError('endpoint theta shape differs')
        weights,cap=projection.mapped_evaluation(theta);saved=np.load(paths[0],allow_pickle=False)
        metadata=dict(job=job['id'],count=job['transitions'],shape=list(weights.shape),
            weight_hash=hashlib.sha256(weights.tobytes()).hexdigest())
        if (not np.array_equal(saved,weights) or json.loads(paths[1].read_text())!=metadata
                or json.loads(paths[3].read_text())!=cap):raise ValueError('projected endpoint artifacts disagree')
    else:paths=[directory/job[k] for k in ('weights_from','scaling_from') if job.get(k) is not None]
    return {p.name:sha(p) for p in [*paths,directory/'projection.json']}

def completed_count(directory,job):
    path=directory/(job['id']+'.json')
    report=json.loads(path.read_text())
    count=sample_count(directory,job['id'])
    if not report['complete'] or report['transitions']!=count:
        raise ValueError('completed report differs from journal row count')
    if job['kind']=='learn' and count!=job['transitions']:
        raise ValueError('completed training must have exactly the declared endpoint rows')
    if job['kind']=='evaluate' and report['episodes']!=job['config']['episodes']:
        raise ValueError('completed evaluation episode count differs')
    return report,count

def completion_payload(directory,job):
    _,count=completed_count(directory,job)
    path=directory/(job['id']+'.json')
    return dict(report_hash=sha(path),journal_hash=journal_hash(directory/(job['id']+'.db')),
                rows=count,endpoint_artifacts=endpoint_artifacts(directory,job))

def pin_completion(directory,job):
    """Call only after the worker has verified its actual durable session."""
    path=directory/(job['id']+'.json')
    if path.exists() and json.loads(path.read_text()).get('complete'):
        atomic_json(directory/(job['id']+'-completion.json'),completion_payload(directory,job))

def complete_report(directory,job):
    pin=directory/(job['id']+'-completion.json')
    if not pin.exists():return None
    if json.loads(pin.read_text())!=completion_payload(directory,job):
        raise ValueError('completed report, journal or endpoint artifacts changed')
    return json.loads((directory/(job['id']+'.json')).read_text())

def prepare(directory):
    directory.mkdir(parents=True,exist_ok=True)
    proof=json.loads((directory/'baseline-projection-proof.json').read_text())
    reference=json.loads((REFERENCE/'manifest.json').read_text())
    source=source_signature();runtime=runtime_signature()
    if (proof['source']!=source or proof['reference_source']!=reference['source']
            or reference['runtime']!=runtime or not all(r['exact'] for r in proof['proof'])):
        raise ValueError('baseline source/action/physics/reset proof differs')
    descriptor=json.loads((directory/'projection.json').read_text())
    projection=FixedPolicyProjection(4389,descriptor)
    if (descriptor['shape']!=[214,4389] or descriptor['max_norm2']!=256.
            or descriptor['file_sha256']!=sha(TRANSFORM)
            or descriptor['calibration']['fit_sha256']!=sha(FIT)
            or descriptor['calibration']['observational_rows']!=1024
            or descriptor['calibration']['no_rewards_or_labels_used'] is not True):
        raise ValueError('exact already frozen generic transform/calibration required')
    projection.validate_embedded()
    paths=[REFERENCE/f'eval_{name}.json' for name in ('initial','up','random','noop')]
    controls=[json.loads(p.read_text()) for p in paths]
    gate=thresholds(controls)
    if gate!=dict(total=14,own_hit=6):raise ValueError('declared control gate differs')
    expected=dict(source=source,runtime=runtime,model_hash=MODEL_HASH,jobs=declared_jobs(),
        driver_hash=sha(WORKER),coordinator_hash=sha(__file__),protocol_hash=sha(PROTOCOL),
        integrity_helper_hash=sha(INTEGRITY_HELPER),
        projection_hash=sha(directory/'projection.json'),matrix_file_hash=sha(TRANSFORM),fit_hash=sha(FIT),
        baseline_proof_hash=sha(directory/'baseline-projection-proof.json'),preflight_hash=sha(PREFLIGHT),
        baseline_report_hashes={p.name:sha(p) for p in paths},gate=gate,
        learning_camera_cap=90000,evaluation_camera_cap=86400,workers=3,training_seeds=list(TRAINING_SEEDS),
        training_wall_cap_seconds=TRAINING_SECONDS,overall_wall_cap_seconds=TOTAL_SECONDS,
        long_bank_entries=0,context_fallback_steps=4,torch_threads=1,
        calibration_rows_reused=1024,new_calibration_observations=0,reserved_final_seeds=list(FINAL))
    path=directory/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if any(manifest[k]!=v for k,v in expected.items()):raise ValueError('changed declared study')
    else:
        manifest=dict(**expected,started_wall_ns=time.time_ns())
        atomic_json(path,manifest)
    return manifest,controls

def supervise(directory):
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest,controls=prepare(directory)
        def execute_checked(job):
            report=directory/(job['id']+'.json');stop=directory/(job['id']+'-registered-stop.json')
            if stop.exists():return json.loads(stop.read_text())
            deadline=manifest['started_wall_ns']/1e9+(TRAINING_SECONDS if job['kind']=='learn' else TOTAL_SECONDS)
            while complete_report(directory,job) is None:
                verification=bool(report.exists() and json.loads(report.read_text()).get('complete'))
                before=sample_count(directory,job['id'])
                # A complete-but-unpinned report must recover the actual session
                # with zero new samples; never let stale JSON trigger new work.
                if verification:completed_count(directory,job)
                remaining=deadline-time.time()
                if remaining<25:
                    return dict(job=job['id'],status='partial',reason='registered wall budget exhausted')
                started=time.monotonic();child=None
                try:
                    child=subprocess.run([sys.executable,str(WORKER),'--directory',str(directory),'--worker',job['id'],
                        '--max-work-seconds',str(min(160.,remaining-25)),'--chunks-per-worker','5','--warm-session'],
                        capture_output=True,text=True,timeout=min(180.,remaining-5))
                    cost=dict(job=job['id'],wall_seconds=time.monotonic()-started,returncode=child.returncode,
                        result=json.loads(child.stdout) if child.returncode==0 else None,stderr=child.stderr[-4000:])
                except subprocess.TimeoutExpired:
                    cost=dict(job=job['id'],wall_seconds=time.monotonic()-started,returncode=None,timeout=True)
                with (directory/(job['id']+'-costs.jsonl')).open('a') as out:
                    out.write(json.dumps(cost)+'\n');out.flush();os.fsync(out.fileno())
                print(json.dumps(cost),flush=True)
                if child is None or child.returncode:
                    failure=dict(job=job['id'],status='failed',reason=cost.get('stderr','timeout'))
                    atomic_json(stop,failure);return failure
                if verification and (sample_count(directory,job['id'])!=before or cost['result']['new_transitions']!=0):
                    raise ValueError('completion verification unexpectedly added observations')
                pin_completion(directory,job)
            return dict(job=job['id'],status='complete')
        def execute(job):
            try:return execute_checked(job)
            except Exception as error:
                failure=dict(job=job['id'],status='failed',reason=f'{type(error).__name__}: {error}')
                atomic_json(directory/(job['id']+'-registered-stop.json'),failure)
                return failure
        with ThreadPoolExecutor(max_workers=3) as pool:
            results=list(pool.map(execute,[j for j in manifest['jobs'] if j['kind']=='learn']))
            done={r['job']:r['status']=='complete' for r in results}
            eligible=[j for seed,j in zip(TRAINING_SEEDS,[j for j in manifest['jobs'] if j['kind']=='evaluate'])
                      if done.get(f'learn_dct_{seed}')]
            results+=list(pool.map(execute,eligible))
        reports={j['id']:json.loads((directory/(j['id']+'.json')).read_text()) for j in manifest['jobs'] if (directory/(j['id']+'.json')).exists()}
        counts=dict(learn=0,evaluate=0)
        for job in manifest['jobs']:
            db=directory/(job['id']+'.db')
            if db.exists():
                with sqlite3.connect(db) as c:counts[job['kind']]+=c.execute('SELECT count(*) FROM replay_samples').fetchone()[0]
        assert counts['learn']<=90000 and counts['evaluate']<=86400
        complete=len(results)==6 and all(r['status']=='complete' for r in results)
        verified={j['id']:complete_report(directory,j) for j in manifest['jobs']
                  if done.get(j['id'],False) or any(r['job']==j['id'] and r['status']=='complete' for r in results)}
        decision=condition_decision(verified,controls);gate=manifest['gate']
        qualifies=bool(complete and decision['qualifies'])
        summary=dict(status='complete' if complete else 'partial',counts=counts,reports=reports,results=results,
            controls=controls,gate=gate,decision=decision,qualifies=qualifies,endpoint_per_seed=30000,
            wall_seconds=time.time()-manifest['started_wall_ns']/1e9,final_holdout_unused=list(FINAL),
            further_replication_or_holdout_not_allocated=True,scope='Score-only .25 screen; no original.45 or forecast-benefit claim.')
        atomic_json(directory/'summary.json',summary)
        print(json.dumps({k:summary[k] for k in ('status','counts','gate','qualifies','wall_seconds')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare(args.directory)
    else:supervise(args.directory)
