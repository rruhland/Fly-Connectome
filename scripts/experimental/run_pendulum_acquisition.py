"""One frozen native-survival/plastic-vision acquisition comparison; explicit run only."""
import argparse,ast,fcntl,json,multiprocessing as mp,os,shutil,time,traceback
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import asdict
from pathlib import Path
import numpy as np
from fly_connectome.control.session import source_signature,MODEL_HASH,VISION_CONFIGURATION
from fly_connectome.control.durable import runtime_signature
from fly_connectome.control.projection import PROJECTION_VERSION
from fly_connectome.control.vision_transfer import bridge_snapshot
from fly_connectome.control.pendulum_evaluation import PendulumEvaluationConfig
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from run_pong_development import atomic_json,FINAL_SEEDS
from run_direct_hit_policy_search import initialize_worker
import online_vision_metrics as pong_metrics
import pendulum_acquisition_metrics as metrics
import pendulum_acquisition_worker as worker

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'docs/reconstruction/2026-10-21-pendulum-acquisition-protocol.md'
ONLINE=Path('/workspace/m1b-experiments/pong-online-vision-drift-20261021-r1')
CAL=Path('/workspace/m1b-experiments/pendulum-preflight-20261021/calibration')
PREFLIGHT=CAL.parent/'plastic-runtime-seed-separated'
PARENT=ONLINE.parent/'online-vision-parent-audit-20261021'
ARMS=('experienced','fresh');DOSE=20000;CHECKPOINTS=(5000,10000,20000)
ROOTS=dict(environment=1836497107,learner=720583127,development=915760327,control_actions=1762054381)
TAG=29114;BOOTSTRAP=1107836249;TRAIN_SECONDS=1200;EVAL_SECONDS=1200;TOTAL_SECONDS=2400
sha=worker.sha


def pinned_sources():
    names=['run_pendulum_acquisition.py','pendulum_acquisition_worker.py','pendulum_acquisition_metrics.py',
        'persistent_vision_worker.py','online_vision_metrics.py','return_metrics.py','run_pong_development.py',
        'run_pong_policy_search_continuation.py','run_direct_hit_policy_search.py']
    return {**{'scripts/experimental/'+name:sha(ROOT/'scripts/experimental'/name) for name in names},
            str(PROTOCOL.relative_to(ROOT)):sha(PROTOCOL)}


def immutable_copy(source,target):
    if target.exists():
        if sha(source)!=sha(target):raise ValueError('fixed input copy changed')
    else:shutil.copyfile(source,target)


def schedule(excluded):
    used=set(excluded)|set(FINAL_SEEDS)|set(ROOTS.values())
    rng=np.random.default_rng(np.random.SeedSequence([ROOTS['development'],TAG]));seeds=[]
    while len(seeds)<64:
        value=int(rng.integers(1,2147483647))
        if value not in used:seeds.append(value);used.add(value)
    actions=[int(np.random.SeedSequence([ROOTS['control_actions'],TAG,i]).generate_state(1)[0]) for i in range(64)]
    if set(actions)&set(FINAL_SEEDS):raise ValueError('reserved control-action seed collision')
    return dict(development=seeds,control_actions=actions)


def stage(directory):
    directory.mkdir(parents=True,exist_ok=True)
    exclusion=directory/'seed-exclusions.json'
    if not exclusion.exists():raise ValueError('reviewed complete prior/diagnostic exclusion snapshot required')
    excluded=json.loads(exclusion.read_text())['ids']
    if set(ROOTS.values())&set(excluded):raise ValueError('new root collision')
    old=json.loads((ONLINE/'manifest.json').read_text());source=source_signature()
    if not all(json.loads(path.read_text())['passed'] for path in
        (ONLINE/'completion-verification.json',PARENT/'summary.json',PREFLIGHT/'report.json')):
        raise ValueError('verified online ancestors and runtime fixture required')
    # The cache has no learner. Only its optional session changed, including RNG
    # separation and plain summary accounting; perception/physics/fit are exact.
    calibration=json.loads((CAL/'manifest.json').read_text())
    delta={name:dict(before=calibration['source'].get(name),after=source.get(name))
        for name in set(source)|set(calibration['source']) if source.get(name)!=calibration['source'].get(name)}
    if set(delta)!={'control/pendulum_session.py'}:raise ValueError('calibration compatibility delta changed')
    tested=json.loads((PREFLIGHT/'report.json').read_text())
    fixture_delta={name for name in set(source)|set(tested['source']) if source.get(name)!=tested['source'].get(name)}
    fixture_source=CAL.parent/'attempt2/source/src/fly_connectome/control/pendulum_session.py'
    def without_summary(path):
        tree=ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node,ast.ClassDef):node.body=[n for n in node.body if not isinstance(n,ast.FunctionDef) or n.name!='summary']
        return ast.dump(tree,include_attributes=False)
    if (fixture_delta!={'control/pendulum_session.py'}
            or sha(fixture_source)!=tested['source']['control/pendulum_session.py']
            or without_summary(fixture_source)!=without_summary(ROOT/'src/fly_connectome/control/pendulum_session.py')):
        raise ValueError('runtime fixture permits only plain summary export delta')
    changes={name:dict(before=old['source'].get(name),after=source.get(name))
        for name in sorted(set(source)|set(old['source'])) if source.get(name)!=old['source'].get(name)}
    expected_changes={'control/codec.py','control/durable.py','control/pendulum.py',
        'control/pendulum_session.py','control/pendulum_evaluation.py','control/vision_transfer.py'}
    if set(changes)!=expected_changes:raise ValueError('reviewed visual-source bridge changed')
    fixed={};bridges={};starts={}
    for arm,name in [('experienced','seed1_exposure-retained.pkl'),('fresh','seed1-before.pkl')]:
        digest=(json.loads((ONLINE/'seed1_exposure.json').read_text())['retained_snapshot']['file_sha256']
            if arm=='experienced' else old['fixed_artifacts'][name])
        snapshot,receipt=bridge_snapshot(ONLINE/name,expected_file_sha256=digest,
            ancestor_source=old['source'],reviewed_control_changes=changes)
        saved=worker.save_snapshot(directory/(arm+'-before.pkl'),snapshot)
        atomic_json(directory/(arm+'-bridge.json'),receipt);bridges[arm]=receipt;starts[arm]=saved
        fixed[saved['file']]=saved['file_sha256']
    for name in ('search-transform.npy','transform-fit.json','generic-features.npz'):
        immutable_copy(CAL/name,directory/name);fixed[name]=sha(directory/name)
    for name in ('seed1-policy_endpoint.npy','seed1-radial-cap.json'):
        immutable_copy(ONLINE/name,directory/name);fixed[name]=sha(directory/name)
    matrix=np.load(directory/'search-transform.npy',allow_pickle=False)
    projection=dict(version=PROJECTION_VERSION,matrix_file=str(directory/'search-transform.npy'),
        file_sha256=sha(directory/'search-transform.npy'),array_sha256=__import__('hashlib').sha256(matrix.tobytes()).hexdigest(),
        shape=list(matrix.shape),max_norm2=256.,calibration=dict(score_free=True,visual_learning=False,
            cache_sha256=sha(directory/'generic-features.npz'),fit_sha256=sha(directory/'transform-fit.json'),
            compatibility_delta=delta))
    pins={str(path):sha(path) for path in (ONLINE/'manifest.json',ONLINE/'summary.json',
        ONLINE/'return-annotations.json',ONLINE/'completion-verification.json',PARENT/'summary.json',
        CAL/'manifest.json',CAL/'report.json',PREFLIGHT/'manifest.json',PREFLIGHT/'report.json')}
    expected=dict(version='native-survival-plastic-vision-pair-v1',source=source,runtime=runtime_signature(),
        model_hash=MODEL_HASH,vision_configuration=VISION_CONFIGURATION,pinned_sources=pinned_sources(),
        seed_roots=ROOTS,seed_tag=TAG,schedules=schedule(excluded),bootstrap_seed=BOOTSTRAP,
        seed_exclusions_sha256=sha(exclusion),fixed_artifacts=fixed,ancestor_evidence=pins,starts=starts,
        source_bridges=bridges,calibration_compatibility_delta=delta,projection=projection,
        runtime_fixture_compatibility='Exact AST after excluding new plain summary method; no execution/learning delta',
        transitions_per_arm=DOSE,checkpoints=list(CHECKPOINTS),training_seconds=TRAIN_SECONDS,
        evaluation_seconds=EVAL_SECONDS,total_seconds=TOTAL_SECONDS,probe_serves=64,
        evaluation_priority='B endpoints+five shared controls; A after-B; descriptive B midpoints',
        a_reused_development_seeds=old['schedules']['probes'],
        a_before_groups=dict(experienced='seed1_after',fresh='seed1_before'),
        reserved_final_ids=list(FINAL_SEEDS),final_streams_expanded=False)
    path=directory/'manifest.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if any(saved.get(k)!=v for k,v in expected.items()):raise ValueError('staged pair changed')
    else:atomic_json(path,dict(**expected,jobs=[]))


def validate_manifest(directory):
    m=json.loads((directory/'manifest.json').read_text())
    if (m['source']!=source_signature() or m['runtime']!=runtime_signature()
            or m['pinned_sources']!=pinned_sources() or m['seed_roots']!=ROOTS
            or sha(directory/'seed-exclusions.json')!=m['seed_exclusions_sha256']
            or m['schedules']!=schedule(json.loads((directory/'seed-exclusions.json').read_text())['ids'])):
        raise ValueError('registered pair source/runtime/seed evidence changed')
    if any(sha(directory/name)!=digest for name,digest in m['fixed_artifacts'].items()):
        raise ValueError('fixed pair input changed')
    if any(sha(Path(name))!=digest for name,digest in m['ancestor_evidence'].items()):
        raise ValueError('reused A/calibration/runtime evidence changed')
    return m


def register_jobs(directory,m,jobs):
    saved={job['id']:job for job in m['jobs']}
    for job in jobs:
        if job['id'] in saved and saved[job['id']]!=job:raise ValueError('job definition changed')
        path=directory/(job['id']+'-spec.json')
        if path.exists() and json.loads(path.read_text())!=job:raise ValueError('job spec changed')
        atomic_json(path,job);saved[job['id']]=job
    m['jobs']=list(saved.values());atomic_json(directory/'manifest.json',m)


def training_jobs(m):
    return [dict(id=arm+'_train',arm=arm,kind='train',phase='training',camera_dose=DOSE,
        checkpoints=list(CHECKPOINTS),snapshot_from=m['starts'][arm]['file'],
        input_hashes={name:m['fixed_artifacts'][name] for name in (m['starts'][arm]['file'],'search-transform.npy')},
        config=dict(seed=ROOTS['environment'],learner_seed=ROOTS['learner'],max_episode_ticks=600,
            alpha=.001,lam=.995,epsilon=.2,gamma_per_second=.95)) for arm in ARMS]


def evaluation_jobs(directory,m):
    jobs=[]
    def job(name,kind,group,i,seed,snapshot,weights,config):
        refs=[snapshot]+([] if weights is None else [weights])
        return dict(id=name,kind=kind,phase='evaluation',group=group,serve_index=i,
            config=asdict(config),snapshot_from=snapshot,weights_from=weights,
            input_hashes={name:sha(directory/name) for name in refs},
            **(dict(learn_enabled=False,episode_seeds=[seed]) if kind=='pong' else {}))
    for arm in ARMS:
        for i,seed in enumerate(m['schedules']['development']):
            prefix=arm+'_at20000';cap=json.loads((directory/(prefix+'-cap.json')).read_text())
            config=PendulumEvaluationConfig(seed,policy_scaling=cap)
            jobs.append(job(f'b_{arm}_endpoint_s{i:02d}','pendulum',arm+'_endpoint',i,seed,
                prefix+'-vision.pkl',prefix+'-weights.npy',config))
    for kind in metrics.CONTROLS:
        for i,seed in enumerate(m['schedules']['development']):
            config=PendulumEvaluationConfig(seed,policy_kind='random' if kind.startswith('random') else kind,
                action_seed=m['schedules']['control_actions'][i] if kind.startswith('random') else None,
                skip_baseline_perception=True,action_hold_frames=10 if kind=='random10' else 1)
            jobs.append(job(f'b_control_{kind}_s{i:02d}','pendulum',kind,i,seed,'fresh-before.pkl',None,config))
    cap=json.loads((directory/'seed1-radial-cap.json').read_text())
    for arm in ARMS:
        for i,seed in enumerate(m['a_reused_development_seeds']):
            config=PongEvaluationConfig(seed,episodes=1,max_episode_ticks=1200,opponent_speed=.65,policy_scaling=cap)
            jobs.append(job(f'a_{arm}_s{i:02d}','pong',arm+'_a_after',i,seed,arm+'_at20000-vision.pkl',
                'seed1-policy_endpoint.npy',config))
    for dose in (5000,10000):
        for arm in ARMS:
            prefix=f'{arm}_at{dose}';cap=json.loads((directory/(prefix+'-cap.json')).read_text())
            for i,seed in enumerate(m['schedules']['development'][:32]):
                jobs.append(job(f'b_{arm}_at{dose}_s{i:02d}','pendulum',prefix,i,seed,prefix+'-vision.pkl',
                    prefix+'-weights.npy',PendulumEvaluationConfig(seed,policy_scaling=cap)))
    return jobs


def execute_phase(directory,m,phase,jobs,workers,*,phase_seconds=None,total_seconds=None):
    path=directory/(phase+'-boundary.json')
    start=m['started_wall_ns'] if phase=='training' else json.loads((directory/'training-boundary.json').read_text())['frozen_wall_ns']
    duration=(TRAIN_SECONDS if phase=='training' else EVAL_SECONDS) if phase_seconds is None else phase_seconds
    total=TOTAL_SECONDS if total_seconds is None else total_seconds
    if duration<=0 or total<=0:raise ValueError('positive phase and total wall caps required')
    expected=dict(started_wall_ns=start,deadline_wall_ns=min(start+int(duration*1e9),m['started_wall_ns']+int(total*1e9)))
    costs=directory/(phase+'-costs.jsonl');failed=costs.exists() and any(not json.loads(x)['ok'] for x in costs.read_text().splitlines())
    if path.exists():
        b=json.loads(path.read_text())
        if any(b[k]!=v for k,v in expected.items()):raise ValueError('phase clock changed')
        if 'frozen_wall_ns' in b:
            if b['complete']!=all(worker.complete_report(directory,j) is not None for j in jobs) or b['failed']!=failed:
                raise ValueError('frozen phase completion changed')
            return b
    else:atomic_json(path,expected)
    deadline=expected['deadline_wall_ns']/1e9;queue=[j for j in jobs if worker.complete_report(directory,j) is None]
    # Submission order is the prespecified priority; at most one worker owns a job.
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize_worker) as pool:
        pending={}
        while pending or queue:
            while queue and len(pending)<workers and time.time()<deadline and not failed:
                j=queue.pop(0);pending[pool.submit(worker.execute_job,str(directory),j,deadline)]=(j,time.monotonic())
            if not pending:break
            done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
            for future in done:
                j,submitted=pending.pop(future)
                try:cost=future.result()
                except BaseException:cost=dict(ok=False,job=j['id'],traceback=traceback.format_exc(),
                    wall_seconds=time.monotonic()-submitted,wall_definition='Parent submit-to-failure including worker startup/queue')
                failed=failed or not cost['ok']
                with costs.open('a') as f:f.write(json.dumps(cost)+'\n');f.flush();os.fsync(f.fileno())
    b=dict(**expected,frozen_wall_ns=time.time_ns(),failed=bool(failed),
        complete=all(worker.complete_report(directory,j) is not None for j in jobs))
    b['budget_met']=b['frozen_wall_ns']<=b['deadline_wall_ns'];atomic_json(path,b);return b


def outcome_evidence(directory,m):
    files=['manifest.json','cost-summary.json']
    files += [name for name in ('training-boundary.json','evaluation-boundary.json','a-after-annotations.json') if (directory/name).exists()]
    return dict(files={name:sha(directory/name) for name in files},jobs={j['id']:dict(
        journal_hash=worker.journal_hash(directory/(j['id']+'.db')) if (directory/(j['id']+'.db')).exists() else None,
        files={suffix:sha(directory/(j['id']+suffix)) for suffix in ('.json','-completion.json','-spec.json')
            if (directory/(j['id']+suffix)).exists()}) for j in m['jobs']})


def retained_outcome(directory,m):
    path=directory/'study-completion.json'
    if not path.exists():
        if (directory/'summary.json').exists():raise ValueError('public summary lacks authoritative completion')
        return None
    saved=json.loads(path.read_text())
    if saved['evidence']!=outcome_evidence(directory,m):raise ValueError('retained pair evidence changed')
    if (directory/'summary.json').exists() and json.loads((directory/'summary.json').read_text())!=saved['summary']:
        raise ValueError('retained pair summary changed')
    atomic_json(directory/'summary.json',saved['summary']);return saved['summary']


def supervise(directory):
    with (directory/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);m=validate_manifest(directory)
        old=retained_outcome(directory,m)
        if old is not None:return old
        if 'started_wall_ns' not in m:m['started_wall_ns']=time.time_ns();atomic_json(directory/'manifest.json',m)
        train=training_jobs(m);register_jobs(directory,m,train)
        tb=execute_phase(directory,m,'training',train,2);eb=None;groups={};a_groups={};annotations={}
        if tb['complete'] and not tb['failed']:
            jobs=evaluation_jobs(directory,m);register_jobs(directory,m,jobs)
            eb=execute_phase(directory,m,'evaluation',jobs,4)
            for j in jobs:
                report=worker.complete_report(directory,j)
                if j['kind']=='pong':
                    a=pong_metrics.annotate_job(directory,j);annotations[j['id']]=a
                    if report is not None:a_groups.setdefault(j['group'],[]).extend(a['rallies'])
                elif report is not None:groups.setdefault(j['group'],[]).append(dict(report['session'],
                    seed=j['config']['seed'],serve_index=j['serve_index']))
        atomic_json(directory/'a-after-annotations.json',annotations)
        evidence=metrics.survival_evidence(groups,BOOTSTRAP);retention={}
        before=json.loads((ONLINE/'return-annotations.json').read_text())
        for arm in ARMS:
            oldgroup=m['a_before_groups'][arm]
            original=[before[f'{oldgroup}_s{i:02d}']['rallies'][0] for i in range(64)]
            retention[arm]=metrics.retention_evidence(original,a_groups.get(arm+'_a_after',[]),BOOTSTRAP+1)
        costs=cost_summary(directory,m);atomic_json(directory/'cost-summary.json',costs)
        elapsed=(time.time_ns()-m['started_wall_ns'])/1e9
        eval_elapsed=None if eb is None else (time.time_ns()-eb['started_wall_ns'])/1e9
        valid=bool(tb['complete'] and tb['budget_met'] and not tb['failed'] and eb is not None
            and eb['complete'] and eb['budget_met'] and not eb['failed'] and eval_elapsed<=EVAL_SECONDS and elapsed<=TOTAL_SECONDS)
        summary=dict(status='complete' if valid else 'partial',training=tb,evaluation=eb,wall_seconds=elapsed,
            evaluation_scoring_wall_seconds=eval_elapsed,overall_budget_met=elapsed<=TOTAL_SECONDS,
            evidence_eligible=valid,b_survival=evidence,a_retention=retention,costs=costs,
            interpretation='Native reward acquisition with plastic vision; one paired training seed, reused A development probes, no formal retention noninferiority',
            final_streams_expanded=False)
        atomic_json(directory/'study-completion.json',dict(summary=summary,evidence=outcome_evidence(directory,m)))
        atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True);return summary


def cost_summary(directory,m):
    import sqlite3
    logical={};settled=[];latencies={}
    for phase in ('training','evaluation'):
        path=directory/(phase+'-costs.jsonl')
        if path.exists():settled.extend(json.loads(line) for line in path.read_text().splitlines())
    for j in m['jobs']:
        path=directory/(j['id']+'.db')
        if not path.exists():continue
        with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as c:
            rows=c.execute('SELECT record FROM replay_samples ORDER BY seq').fetchall()
            events=c.execute('SELECT kind,count,wall_ns FROM replay_timing ORDER BY id').fetchall()
        values=latencies.setdefault(j['kind'],[])
        values.extend((b[2]-a[2])/1e6 for a,b in zip(events,events[1:])
            if a[0]==b[0]=='heartbeat' and b[1]==a[1]+1)
        data=[json.loads(r[0]) for r in rows]
        if j['kind']=='train':initials=1+sum(r['terminated'] or r['truncated'] for r in data);visual=len(rows)+initials
        else:initials=1;visual=0 if j['config'].get('skip_baseline_perception') else len(rows)+1
        label='training' if j['kind']=='train' else 'a_retention' if j['kind']=='pong' else 'b_evaluation'
        v=logical.setdefault(label,dict(transitions=0,initial_images=0,visual_images=0))
        v['transitions']+=len(rows);v['initial_images']+=initials;v['visual_images']+=visual
    return dict(logical_committed_work=logical,summed_settled_task_wall_seconds=sum(x.get('wall_seconds',0.) for x in settled),
        replayed_camera_transitions=sum(x.get('replayed_camera_transitions',0) for x in settled),
        failed_cost_records=sum(not x['ok'] for x in settled),
        maximum_worker_process_lifetime_peak_rss_kib=max((x.get('process_peak_rss_kib',0) for x in settled),default=0),
        committed_camera_latency_ms={kind:dict(samples=len(v),quantiles=np.quantile(v,[.5,.95,.99,1]).tolist())
            for kind,v in latencies.items() if v},
        latency_definition='Consecutive per-camera heartbeat intervals including snapshots, excluding recovery/process boundaries',
        scoring_physics_reexecutions=sum(len(v.get('rallies',[])) and v['camera_rows'] for v in
            json.loads((directory/'a-after-annotations.json').read_text()).values()),
        caveat='Logical rows include failed/partial jobs and eager reset initials. Constructor failures before a journal and uncommitted crash tails may consume unmeasured compute; never claimed zero.',
        reused_a_before='128 completed old development trials, zero newly collected before-B cameras')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--stage',action='store_true');mode.add_argument('--run',action='store_true')
    args=parser.parse_args();stage(args.directory) if args.stage else supervise(args.directory)
