"""Fixed three-arm reward-credit diagnostic; shared DCT policy and raw-score eval."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import fcntl,json,os,sqlite3,subprocess,sys,time
from pathlib import Path
import run_dct_sarsa_pong as base
from run_pong_development import atomic_json
from fly_connectome.control.session import source_signature,MODEL_HASH
from fly_connectome.control.durable import runtime_signature
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.projection import FixedPolicyProjection
from fly_connectome.control.rewards import REWARD_VERSION

ROOT=Path(__file__).resolve().parents[2];WORKER=base.WORKER
PROTOCOL=ROOT/'docs/reconstruction/2026-10-16-reward-credit-diagnostic-protocol.md'
ARMS=('score','hit','survival');TRAIN_SECONDS=720.;TOTAL_SECONDS=1500.
sha=base.sha

def declared_jobs():
    coefficients=dict(score=(0.,0.),hit=(1.,0.),survival=(0.,.1));jobs=[]
    for arm in ARMS:
        hit,survival=coefficients[arm]
        jobs.append(dict(id=f'learn_{arm}',kind='learn',seed=172900127,alpha=.001,lam=.995,
            opponent_speed=.25,transitions=12000,checkpoints=[12000],projection_from='projection.json',
            reward_shaping=dict(version=REWARD_VERSION,own_hit_bonus=hit,survival_per_second=survival),
            compact_reports=True,max_abs_q=1000.))
    for arm in ARMS:
        jobs.append(dict(id=f'eval_{arm}',kind='evaluate',config=asdict(PongEvaluationConfig(904271833,
            episodes=24,opponent_speed=.25)),weights_from=f'learn_{arm}_at12000.npy',
            scaling_from=f'learn_{arm}_at12000.cap.json',compact_reports=True))
    return jobs

def prepare(directory):
    directory.mkdir(parents=True,exist_ok=True);source=source_signature();runtime=runtime_signature()
    proof=json.loads((directory/'baseline-projection-proof.json').read_text())
    reference=json.loads((base.REFERENCE/'manifest.json').read_text())
    if (proof['source']!=source or proof['reference_source']!=reference['source']
            or runtime!=reference['runtime'] or not all(p['exact'] for p in proof['proof'])):
        raise ValueError('matched baseline source/physics/reset proof required')
    descriptor=json.loads((directory/'projection.json').read_text())
    projection=FixedPolicyProjection(4389,descriptor);projection.validate_embedded()
    if (descriptor['shape']!=[214,4389] or descriptor['max_norm2']!=256.
            or descriptor['file_sha256']!=sha(base.TRANSFORM)
            or descriptor['calibration']['fit_sha256']!=sha(base.FIT)):
        raise ValueError('exact frozen observational DCT transform and cap required')
    paths=[base.REFERENCE/f'eval_{name}.json' for name in ('initial','up','random','noop')]
    controls=[json.loads(p.read_text()) for p in paths]
    expected=dict(source=source,runtime=runtime,model_hash=MODEL_HASH,jobs=declared_jobs(),
        driver_hash=sha(WORKER),coordinator_hash=sha(__file__),completion_helper_hash=sha(base.__file__),
        semantic_helper_hash=sha(base.INTEGRITY_HELPER),protocol_hash=sha(PROTOCOL),
        projection_hash=sha(directory/'projection.json'),baseline_proof_hash=sha(directory/'baseline-projection-proof.json'),
        control_report_hashes={p.name:sha(p) for p in paths},gate=base.thresholds(controls),
        training_camera_cap=36000,evaluation_camera_cap=86400,train_wall_seconds=TRAIN_SECONDS,
        overall_wall_seconds=TOTAL_SECONDS,workers=3,long_bank_entries=0,context_fallback_steps=4,
        calibration_rows_reused=1024,new_calibration_observations=0,reserved_final_seeds=list(base.FINAL))
    path=directory/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if any(manifest[k]!=v for k,v in expected.items()):raise ValueError('registered diagnostic changed')
    else:
        manifest=dict(**expected,started_wall_ns=time.time_ns());atomic_json(path,manifest)
    return manifest,controls

def supervise(directory):
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);manifest,controls=prepare(directory)
        def execute(job):
            try:
                stop=directory/(job['id']+'-registered-stop.json')
                if stop.exists():return json.loads(stop.read_text())
                report=directory/(job['id']+'.json')
                deadline=manifest['started_wall_ns']/1e9+(TRAIN_SECONDS if job['kind']=='learn' else TOTAL_SECONDS)
                while base.complete_report(directory,job) is None:
                    verification=bool(report.exists() and json.loads(report.read_text()).get('complete'))
                    before=base.sample_count(directory,job['id'])
                    if verification:base.completed_count(directory,job)
                    remaining=deadline-time.time()
                    if remaining<25:return dict(job=job['id'],status='partial',reason='registered wall budget exhausted')
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
                    if child is None or child.returncode:raise RuntimeError(cost.get('stderr','subprocess timeout'))
                    if verification and (base.sample_count(directory,job['id'])!=before or cost['result']['new_transitions']!=0):
                        raise ValueError('completion verification added observations')
                    base.pin_completion(directory,job)
                return dict(job=job['id'],status='complete')
            except Exception as error:
                failure=dict(job=job['id'],status='failed',reason=f'{type(error).__name__}: {error}')
                atomic_json(directory/(job['id']+'-registered-stop.json'),failure);return failure
        with ThreadPoolExecutor(max_workers=3) as pool:
            results=list(pool.map(execute,manifest['jobs'][:3]));done={r['job']:r['status']=='complete' for r in results}
            eligible=[j for arm,j in zip(ARMS,manifest['jobs'][3:]) if done[f'learn_{arm}']]
            results+=list(pool.map(execute,eligible))
        reports={j['id']:json.loads((directory/(j['id']+'.json')).read_text()) for j in manifest['jobs'] if (directory/(j['id']+'.json')).exists()}
        counts=dict(learn=0,evaluate=0)
        for j in manifest['jobs']:counts[j['kind']]+=base.sample_count(directory,j['id'])
        assert counts['learn']<=36000 and counts['evaluate']<=86400
        complete=len(results)==6 and all(r['status']=='complete' for r in results)
        summary=dict(status='complete' if complete else 'partial',counts=counts,reports=reports,results=results,
            controls=controls,gate=manifest['gate'],wall_seconds=time.time()-manifest['started_wall_ns']/1e9,
            no_selection_or_replication=True,final_holdout_unused=list(base.FINAL),
            scope='One-seed reward mechanism screen; raw-score-only frozen evaluation; no acquisition acceptance.')
        atomic_json(directory/'summary.json',summary)
        print(json.dumps({k:summary[k] for k in ('status','counts','wall_seconds')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True,type=Path)
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare(args.directory)
    else:supervise(args.directory)
