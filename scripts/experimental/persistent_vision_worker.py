"""Bounded jobs for the unallocated fixed-policy visual-learning integration study.

Importing this module does not construct a model, expand seeds or start a clock.
The coordinator must supply an already pinned manifest, job and absolute deadline.
"""
import hashlib,json,os,pickle,resource,sqlite3,time
from pathlib import Path
import numpy as np

from fly_connectome.control.codec import fingerprint
from fly_connectome.control.durable import DurableRun,runtime_signature
from fly_connectome.control.online_vision import PersistentVisionSession,VisionSnapshot
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.session import source_signature,MODEL_HASH,VISION_CONFIGURATION
from run_pong_development import atomic_json,checkpoint_open_run
from run_pong_policy_search_continuation import journal_hash

ROOT=Path(__file__).resolve().parents[2]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owned_file(directory,name):
    path=(directory/name).resolve()
    if not path.is_relative_to(directory.resolve()):raise ValueError('job artifact escapes allocation')
    return path


def validate_inputs(directory,job):
    if not isinstance(job.get('id'),str) or Path(job['id']).name!=job['id']:
        raise ValueError('simple persistent job identifier required')
    manifest=json.loads((directory/'manifest.json').read_text())
    if (manifest['source']!=source_signature() or manifest['runtime']!=runtime_signature()
            or manifest['model_hash']!=MODEL_HASH or manifest['vision_configuration']!=VISION_CONFIGURATION):
        raise ValueError('persistent job source/runtime/prior changed')
    if sha(__file__)!=manifest['worker_sha256']:
        raise ValueError('persistent worker source changed')
    for name,digest in manifest['pinned_sources'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha(path)!=digest:
            raise ValueError('persistent coordinator/helper/protocol changed')
    saved=json.loads((directory/(job['id']+'-spec.json')).read_text())
    if saved!=job:raise ValueError('persistent job definition changed')
    if next((j for j in manifest['jobs'] if j['id']==job['id']),None)!=job:
        raise ValueError('persistent job not in registered manifest')
    for name,digest in job['input_hashes'].items():
        if sha(owned_file(directory,name))!=digest:raise ValueError('persistent input artifact changed')
    required=[job['weights_from']]+([] if job['snapshot_from'] is None else [job['snapshot_from']])
    if any(name not in job['input_hashes'] for name in required):
        raise ValueError('readout and starting snapshot must be content pinned')
    config=PongEvaluationConfig(**job['config']);config.validate()
    if (type(job['learn_enabled']) is not bool or not job['episode_seeds']
            or config.episodes!=len(job['episode_seeds']) or config.seed!=job['episode_seeds'][0]
            or config.policy_kind!='trained' or config.blank_camera or config.skip_baseline_perception
            or config.action_hold_frames!=1):
        raise ValueError('fixed readout/live camera and explicit seed schedule required')
    dose=job['camera_dose']
    if job['purpose']=='probe':
        if config.episodes!=1 or job['learn_enabled'] or dose is not None:
            raise ValueError('frozen probe must be exactly one complete rally')
    elif job['purpose'] in ('exposure','prequential'):
        if not job['learn_enabled'] or type(dose) is not int or dose<1:
            raise ValueError('persistent stream needs a positive fixed camera dose')
    else:raise ValueError('unknown persistent job purpose')
    return config


def load_snapshot(directory,name):
    if name is None:return None
    with owned_file(directory,name).open('rb') as f:snapshot=pickle.load(f)
    if type(snapshot) is not VisionSnapshot:raise ValueError('owned visual snapshot required')
    return snapshot.validate()


def save_snapshot(path,snapshot):
    """Export only the exact live dose boundary; identical retries are harmless."""
    snapshot.validate();raw=pickle.dumps(snapshot,protocol=5)
    if path.exists():
        previous=pickle.loads(path.read_bytes()).validate()
        if fingerprint(previous)!=fingerprint(snapshot):raise ValueError('retained boundary snapshot changed')
    else:
        temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        temporary.replace(path)
    return dict(file=path.name,file_sha256=sha(path),content_hash=snapshot.content_hash,
                cumulative_learning_observations=snapshot.learning_observations)


def dose_complete(session,job):
    if job['camera_dose'] is None:return session.complete
    if session.transitions>job['camera_dose']:raise ValueError('camera dose exceeded')
    return session.transitions==job['camera_dose']


def validate_session(session,job,weights,snapshot):
    config=PongEvaluationConfig(**job['config'])
    if (type(session) is not PersistentVisionSession or session.config!=config
            or session.learn_enabled!=job['learn_enabled']
            or session.episode_seeds!=tuple(job['episode_seeds'])
            or not np.array_equal(session.policy.weights,weights)
            or session.starting_visual_snapshot!=(None if snapshot is None else snapshot.content_hash)
            or session.starting_visual_learning_observations!=(0 if snapshot is None else snapshot.learning_observations)):
        raise ValueError('restored persistent session differs from pinned job')


def pin_payload(directory,job):
    validate_inputs(directory,job)
    report_path=directory/(job['id']+'.json');report=json.loads(report_path.read_text())
    db=directory/(job['id']+'.db')
    with sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True) as c:
        rows=c.execute('SELECT count(*) FROM replay_samples').fetchone()[0]
    if not report['complete'] or rows!=report['session']['transitions']:
        raise ValueError('dose completion differs from actual journal rows')
    if job['camera_dose'] is not None and rows!=job['camera_dose']:
        raise ValueError('completed stream has wrong camera dose')
    if job['camera_dose'] is None and not report['session']['complete']:
        raise ValueError('completed frozen probe has unfinished rally')
    outputs=report['retained_snapshot']
    if outputs is not None:
        snapshot=load_snapshot(directory,outputs['file'])
        if (sha(directory/outputs['file'])!=outputs['file_sha256']
                or snapshot.content_hash!=outputs['content_hash']
                or snapshot.learning_observations!=outputs['cumulative_learning_observations']):
            raise ValueError('retained visual endpoint changed')
    return dict(report_sha256=sha(report_path),job_sha256=sha(directory/(job['id']+'-spec.json')),
                journal_hash=journal_hash(db),rows=rows,input_hashes=job['input_hashes'],outputs=outputs)


def complete_report(directory,job):
    path=directory/(job['id']+'-completion.json')
    if not path.exists():return None
    validate_inputs(directory,job)
    if json.loads(path.read_text())!=pin_payload(directory,job):
        raise ValueError('completed persistent evidence changed')
    return json.loads((directory/(job['id']+'.json')).read_text())


def execute_job(directory,job,deadline):
    """One healthy writer per job, camera commits unchanged, absolute wall bound."""
    directory=Path(directory);started=time.monotonic();run=None
    config=validate_inputs(directory,job)
    boundary=json.loads((directory/(job['phase']+'-boundary.json')).read_text())
    if deadline!=boundary['deadline_wall_ns']/1e9:
        raise ValueError('persistent worker deadline differs from durable phase')
    weights=np.load(owned_file(directory,job['weights_from']),allow_pickle=False)
    snapshot=load_snapshot(directory,job['snapshot_from'])
    db=directory/(job['id']+'.db')
    try:
        if db.exists():run=DurableRun(db)
        else:
            session=PersistentVisionSession(weights,config,learn_enabled=job['learn_enabled'],
                episode_seeds=job['episode_seeds'],vision_snapshot=snapshot)
            run=DurableRun(db,session=session)
        validate_session(run.session,job,weights,snapshot)
        initial_count=run.session.transitions;replayed=run.replayed
        initial_images=run.session.initial_cameras;initial_visual=run.session.visual_observations
        if replayed:
            checkpoint=run.last_snapshot_count
            previous=run.db.execute('SELECT record FROM replay_samples WHERE seq=?',(checkpoint-1,)).fetchone() if checkpoint else None
            checkpoint_visual=0 if previous is None else json.loads(previous[0])['visual_learning']['next_observations']
            replayed_visual=initial_visual-checkpoint_visual
        else:replayed_visual=0
        while not dose_complete(run.session,job) and not run.session.complete and time.time()<deadline:
            run.step()
            if run.session.transitions%200==0:
                validate_inputs(directory,job);checkpoint_open_run(run)
                progress=dict(
                    transitions=run.session.transitions,episodes=run.session.episodes,
                    summary=run.session.summary(),process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    current_resident_bytes=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'))
                atomic_json(directory/(job['id']+'-progress.json'),progress)
                with (directory/(job['id']+'-progress.jsonl')).open('a') as f:
                    f.write(json.dumps(progress)+'\n');f.flush();os.fsync(f.fileno())
        done=dose_complete(run.session,job)
        # Closing commits the exact last camera; exporting a scene-reset COPY
        # never removes pending credit or final-image updates from this journal.
        summary=run.session.summary();retained=None
        run.stop()
        validate_inputs(directory,job)
        if job['purpose'] in ('exposure','prequential') and done:
            retained=save_snapshot(directory/(job['id']+'-retained.pkl'),run.session.retained_vision())
        partial=run.session.initial_cameras>run.session.episodes
        with sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True) as c:
            rows=[json.loads(row[0]) for row in c.execute('SELECT record FROM replay_samples ORDER BY seq')]
        raw_hits=sum(row['hits'] for row in rows)
        assert raw_hits==summary['hits']+(run.session.episode_hits if partial else 0)
        report=dict(complete=done,purpose=job['purpose'],session=summary,
            episode_session_complete=summary['complete'],partial_final_episode=partial,
            raw_hits_including_partial_episode=raw_hits,
            raw_reward_including_partial_episode=sum(row['reward'] for row in rows),
            raw_physics_seconds=sum(row['elapsed_seconds'] for row in rows),
            partial_episode=None if not partial else dict(episode=run.session.episodes,
                hits=run.session.episode_hits,seconds=run.session.episode_seconds,reward=run.session.episode_return,
                terminated=False,truncated=False,first_encounter_status='requires posthoc physics annotation'),
            camera_dose=job['camera_dose'],retained_snapshot=retained,
            stopped_reason='dose_complete' if done and job['camera_dose'] is not None else
                'rally_complete' if done else 'seed_schedule_exhausted' if run.session.complete else 'phase_wall_limit',
            fixed_policy_weight_hash=fingerprint(run.session.policy.weights))
        atomic_json(directory/(job['id']+'.json'),report)
        if done:atomic_json(directory/(job['id']+'-completion.json'),pin_payload(directory,job))
        return dict(ok=True,job=job['id'],complete=done,new_camera_transitions=summary['transitions']-initial_count,
                    new_initial_images=summary['initial_cameras']-initial_images,
                    new_visual_observations=summary['visual_observations']-initial_visual,
                    replayed_camera_transitions=replayed,replayed_visual_observations=replayed_visual,
                    replayed_initial_images=replayed_visual-replayed,wall_seconds=time.monotonic()-started,
                    process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except BaseException:
        if run is not None and not run.stopped:run.close_without_checkpoint()
        raise
