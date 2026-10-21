"""Camera-durable jobs for the prospectively registered native-survival pair."""
import json,os,resource,sqlite3,time,traceback
from pathlib import Path
import numpy as np
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.durable import DurableRun,runtime_signature
from fly_connectome.control.pendulum_session import PendulumLearningSession
from fly_connectome.control.pendulum_evaluation import FreshPendulumEvaluator,PendulumEvaluationConfig
from fly_connectome.control.online_vision import PersistentVisionSession
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.session import source_signature
from run_pong_development import atomic_json,checkpoint_open_run
from persistent_vision_worker import sha,owned_file,load_snapshot,save_snapshot
from run_pong_policy_search_continuation import journal_hash

ROOT=Path(__file__).resolve().parents[2]


def validate_inputs(directory,job):
    m=json.loads((directory/'manifest.json').read_text())
    if m['source']!=source_signature() or m['runtime']!=runtime_signature():
        raise ValueError('pendulum allocation source/runtime changed')
    for name,digest in m['pinned_sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('pendulum helper/protocol changed')
    if (next((j for j in m['jobs'] if j['id']==job['id']),None)!=job
            or json.loads((directory/(job['id']+'-spec.json')).read_text())!=job):
        raise ValueError('registered pendulum job changed')
    for name,digest in job['input_hashes'].items():
        if sha(owned_file(directory,name))!=digest:raise ValueError('pendulum input changed')
    required=[job['snapshot_from']]
    if job['kind']!='train' and job.get('weights_from'):required.append(job['weights_from'])
    if job['kind']=='train':required.append('search-transform.npy')
    if job.get('initial_weights_from') is not None:required.append(job['initial_weights_from'])
    if any(name not in job['input_hashes'] for name in required):
        raise ValueError('referenced pendulum inputs must be pinned')
    return m


def save_array(path,array):
    if path.exists():
        if not np.array_equal(np.load(path,allow_pickle=False),array):raise ValueError('boundary array changed')
    else:
        temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as f:np.save(f,array,allow_pickle=False);f.flush();os.fsync(f.fileno())
        temporary.replace(path)


def export_boundary(directory,job,session):
    name=f"{job['arm']}_at{session.transitions}"
    weights,cap=session.mapped_evaluation()
    save_array(directory/(name+'-theta.npy'),session.learner.weights)
    save_array(directory/(name+'-weights.npy'),weights)
    path=directory/(name+'-cap.json')
    if path.exists() and json.loads(path.read_text())!=cap:raise ValueError('boundary cap changed')
    atomic_json(path,cap)
    retained=save_snapshot(directory/(name+'-vision.pkl'),session.retained_vision())
    return dict(transitions=session.transitions,session_hash=fingerprint(session),snapshot=retained,
        files={name+suffix:sha(directory/(name+suffix)) for suffix in
               ('-theta.npy','-weights.npy','-cap.json','-vision.pkl')},summary=session.summary())


def pin_payload(directory,job):
    validate_inputs(directory,job)
    report=json.loads((directory/(job['id']+'.json')).read_text());db=directory/(job['id']+'.db')
    with sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True) as c:
        rows=c.execute('SELECT count(*) FROM replay_samples').fetchone()[0]
        last=c.execute('SELECT record FROM replay_samples ORDER BY seq DESC LIMIT 1').fetchone()
    if not report['complete'] or rows!=report['session']['transitions']:
        raise ValueError('complete report differs from journal rows')
    if job['kind']=='train':
        if rows!=job['camera_dose'] or report['numerical_abort']:raise ValueError('invalid completed training dose')
        if set(report['exports'])!={str(n) for n in job['checkpoints']}:
            raise ValueError('missing prescribed training boundary')
        for export in report['exports'].values():
            if any(sha(directory/name)!=digest for name,digest in export['files'].items()):
                raise ValueError('training boundary artifact changed')
            load_snapshot(directory,export['snapshot']['file'])
    elif (not report['session']['complete'] or last is None
            or not any(json.loads(last[0])[key] for key in ('terminated','truncated'))):
        raise ValueError('completed frozen trial lacks actual ending boundary')
    return dict(report_sha256=sha(directory/(job['id']+'.json')),rows=rows,
        journal_hash=journal_hash(db),input_hashes=job['input_hashes'])


def complete_report(directory,job):
    path=directory/(job['id']+'-completion.json')
    if not path.exists():return None
    if json.loads(path.read_text())!=pin_payload(directory,job):raise ValueError('completed job evidence changed')
    return json.loads((directory/(job['id']+'.json')).read_text())


def execute_job(directory,job,deadline):
    directory=Path(directory);started=time.monotonic();run=None
    m=validate_inputs(directory,job);snapshot=load_snapshot(directory,job['snapshot_from'])
    phase=json.loads((directory/(job['phase']+'-boundary.json')).read_text())
    if deadline!=phase['deadline_wall_ns']/1e9:raise ValueError('absolute job deadline changed')
    db=directory/(job['id']+'.db');created=not db.exists()
    try:
        if not created:run=DurableRun(db)
        else:
            if job['kind']=='train':
                optional={}
                if job.get('initial_weights_from') is not None:
                    optional['weights']=np.load(owned_file(directory,job['initial_weights_from']),allow_pickle=False)
                if 'learn_enabled' in job:optional['learn_enabled']=job['learn_enabled']
                session=PendulumLearningSession(**job['config'],policy_projection=m['projection'],vision_snapshot=snapshot,**optional)
            elif job['kind']=='pendulum':
                weights=None if job['weights_from'] is None else np.load(directory/job['weights_from'],allow_pickle=False)
                session=FreshPendulumEvaluator(weights,PendulumEvaluationConfig.from_dict(job['config']),vision_snapshot=snapshot)
            else:
                config=PongEvaluationConfig(**job['config'])
                session=PersistentVisionSession(np.load(directory/job['weights_from'],allow_pickle=False),config,
                    learn_enabled=False,episode_seeds=[config.seed],vision_snapshot=snapshot)
            run=DurableRun(db,session=session)
        s=run.session
        if (s.starting_visual_snapshot!=snapshot.content_hash
                or s.starting_visual_learning_observations!=snapshot.learning_observations):
            raise ValueError('restored starting visual ancestry changed')
        if job['kind']=='train':
            if any(s.config.get(k)!=v for k,v in job['config'].items()) or s.projection.config!=m['projection']:
                raise ValueError('restored training parameters changed')
            if (getattr(s,'learn_enabled',True)!=job.get('learn_enabled',True)
                    or s.config['visual_learning']!=job.get('learn_enabled',True)):
                raise ValueError('restored visual learning mode changed')
            if job.get('initial_weights_from') is not None:
                initial=np.load(owned_file(directory,job['initial_weights_from']),allow_pickle=False)
                if s.config.get('starting_learner_weights_sha256')!=__import__('hashlib').sha256(initial.tobytes()).hexdigest():
                    raise ValueError('restored starting head ancestry changed')
        elif s.config!=(PendulumEvaluationConfig.from_dict(job['config']) if job['kind']=='pendulum'
                         else PongEvaluationConfig(**job['config'])):
            raise ValueError('restored evaluation parameters changed')
        if job['kind']!='train' and job.get('weights_from') is not None:
            if not np.array_equal(s.policy.weights,np.load(directory/job['weights_from'],allow_pickle=False)):
                raise ValueError('restored frozen policy differs from pinned readout')
        initial=s.transitions;initial_images=s.initial_cameras;initial_visual=s.visual_observations
        replayed=run.replayed;exports={};aborted=False
        if job['kind']=='train':
            for n in job['checkpoints']:
                path=directory/(f"{job['arm']}_at{n}-export.json")
                if path.exists():exports[str(n)]=json.loads(path.read_text())
                elif s.transitions>n:raise ValueError('cannot reconstruct a missing past checkpoint')
        def done():return s.transitions==job['camera_dose'] if job['kind']=='train' else s.complete
        while True:
            if job['kind']=='train' and s.transitions in job['checkpoints']:
                checkpoint_open_run(run);value=export_boundary(directory,job,s)
                old=exports.get(str(s.transitions))
                if old is not None and old!=value:raise ValueError('exported training boundary changed')
                atomic_json(directory/(f"{job['arm']}_at{s.transitions}-export.json"),value)
                exports[str(s.transitions)]=value
            if done() or time.time()>=deadline or aborted:break
            row=run.step()
            if job['kind']=='train':
                aborted=max(abs(row['q']),abs(row['delta']),float(np.max(np.abs(s.learner.weights))))>1000
            if s.transitions%200==0:
                validate_inputs(directory,job);checkpoint_open_run(run)
                progress=dict(session=s.summary(),
                    process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                atomic_json(directory/(job['id']+'-progress.json'),progress)
                with (directory/(job['id']+'-progress.jsonl')).open('a') as f:
                    f.write(json.dumps(progress)+'\n');f.flush();os.fsync(f.fileno())
        summary=s.summary();run.stop();validate_inputs(directory,job)
        report=dict(complete=done() and not aborted,kind=job['kind'],session=summary,exports=exports,
            numerical_abort=aborted,stopped_reason='numerical_abort' if aborted else 'dose_complete' if done()
                and job['kind']=='train' else 'trial_complete' if done() else 'phase_wall_limit')
        atomic_json(directory/(job['id']+'.json'),report)
        if report['complete']:atomic_json(directory/(job['id']+'-completion.json'),pin_payload(directory,job))
        return dict(ok=not aborted,job=job['id'],complete=report['complete'],
            new_camera_transitions=s.transitions-initial,
            new_initial_images=s.initial_cameras-initial_images+int(created),
            new_visual_observations=s.visual_observations-initial_visual+int(created and initial_visual>0),
            replayed_camera_transitions=replayed,wall_seconds=time.monotonic()-started,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except BaseException:
        failure=traceback.format_exc()
        if run is not None and not run.stopped:run.close_without_checkpoint()
        return dict(ok=False,job=job['id'],complete=False,traceback=failure,
            wall_seconds=time.monotonic()-started,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            constructor_started_new_journal=created,
            caveat='Logical committed work is independently counted from the journal; an uncommitted failed step is not reported as zero compute')
