"""Execute the fixed first Pong development allocation with recoverable chunks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from fly_connectome.control.codec import GRAPH_DIGEST
from fly_connectome.control.durable import DurableRun, runtime_signature
from fly_connectome.control.pong_evaluation import FreshPongEvaluator, PongEvaluationConfig, validate_camera_intervention
from fly_connectome.control.pong_actor import FirstHitActorEpisode,validate_bias
from fly_connectome.control.session import LearningSession, source_signature, MODEL_HASH
from fly_connectome.vision import VisualStateEncoder

TRAIN_SEEDS = (172900127,736187003,1597334677)
DEV_SEED = 904271833
FINAL_SEEDS = (308713321,1203078761,1983461927)


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as out:
        json.dump(value,out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
    temporary.replace(path)


def atomic_weights(path,weights):
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as out:
        np.save(out,weights,allow_pickle=False);out.flush();os.fsync(out.fileno())
    temporary.replace(path)


def export_checkpoint(directory,job_id,count,weights,*,checkpoints=(2000,6000)):
    """A boundary commit can survive a crash before its separate weight export."""
    if not checkpoints or list(checkpoints)!=sorted(set(checkpoints)) or any(type(c) is not int or c<1 for c in checkpoints):
        raise ValueError('sorted distinct positive checkpoint counts required')
    for checkpoint in checkpoints:
        path=directory/f'{job_id}_at{checkpoint}.npy'
        if count > checkpoint and not path.exists():
            raise ValueError(f'missing past checkpoint {checkpoint}; reconstruct its exact state')
        if count == checkpoint:
            if path.exists():
                if not np.array_equal(np.load(path,allow_pickle=False),weights):
                    raise ValueError(f'checkpoint {checkpoint} weights disagree with durable state')
            else:
                atomic_weights(path,weights)
            metadata=dict(job=job_id,count=checkpoint,shape=list(weights.shape),
                          weight_hash=hashlib.sha256(weights.tobytes()).hexdigest())
            hash_path=path.with_suffix('.json')
            if hash_path.exists() and json.loads(hash_path.read_text())!=metadata:
                raise ValueError('checkpoint weight metadata disagrees')
            atomic_json(hash_path,metadata)
        if count > checkpoint:
            hash_path=path.with_suffix('.json')
            if not hash_path.exists():raise ValueError('past checkpoint metadata missing')
            saved=np.load(path,allow_pickle=False);metadata=json.loads(hash_path.read_text())
            if list(saved.shape)!=metadata['shape'] or hashlib.sha256(saved.tobytes()).hexdigest()!=metadata['weight_hash']:
                raise ValueError('past checkpoint weights changed')


def export_training_checkpoint(directory,job_id,session,*,checkpoints=(2000,6000)):
    """Keep reduced learner weights and exact mapped evaluation artifacts together.

    A missing export is recoverable only while the journal is still exactly at
    that checkpoint. Past exports fail closed; later weights cannot fill them.
    The default learner retains the existing checkpoint format and behavior.
    """
    projection=getattr(session,'projection',None)
    if projection is None:
        return export_checkpoint(directory,job_id,session.transitions,session.learner.weights,checkpoints=checkpoints)
    count=session.transitions
    for checkpoint in checkpoints:
        if count<=checkpoint:continue
        theta_path=directory/f'{job_id}_at{checkpoint}.theta.npy'
        cap_path=directory/f'{job_id}_at{checkpoint}.cap.json'
        if not theta_path.exists() or not cap_path.exists():
            raise ValueError('missing past projected checkpoint artifacts; reconstruct its exact state')
        mapped,expected=projection.mapped_evaluation(np.load(theta_path,allow_pickle=False))
        if json.loads(cap_path.read_text())!=expected:
            raise ValueError('past projected checkpoint provenance changed')
        metadata=json.loads((directory/f'{job_id}_at{checkpoint}.json').read_text())
        if metadata['weight_hash']!=hashlib.sha256(mapped.tobytes()).hexdigest():
            raise ValueError('past mapped checkpoint differs from projected weights')
    mapped,cap=session.mapped_evaluation() if count in checkpoints else (session.learner.weights,None)
    export_checkpoint(directory,job_id,count,mapped,checkpoints=checkpoints)
    if count in checkpoints:
        theta_path=directory/f'{job_id}_at{count}.theta.npy'
        cap_path=directory/f'{job_id}_at{count}.cap.json'
        if theta_path.exists():
            if not np.array_equal(np.load(theta_path,allow_pickle=False),session.learner.weights):
                raise ValueError('projected checkpoint weights disagree with durable state')
        else:atomic_weights(theta_path,session.learner.weights)
        if cap_path.exists() and json.loads(cap_path.read_text())!=cap:
            raise ValueError('projected checkpoint provenance disagrees with durable state')
        atomic_json(cap_path,cap)


def jobs():
    learning=[];evaluation=[]
    for seed in TRAIN_SEEDS:
        for lam in (.8,.995):
            name=f'learn_{seed}_lambda{lam}'
            learning.append(dict(id=name,kind='learn',seed=seed,lam=lam,transitions=6000))
            for count in (2000,6000):
                evaluation.append(dict(id=f'eval_{seed}_lambda{lam}_at{count}',kind='evaluate',
                    config=asdict(PongEvaluationConfig(DEV_SEED)),weights_from=f'{name}_at{count}.npy'))
    baselines=[]
    for policy in ('initial','random','noop'):
        config=PongEvaluationConfig(DEV_SEED,policy_kind=policy,
            action_seed=447201113 if policy=='random' else None,skip_baseline_perception=True)
        baselines.append(dict(id=f'eval_{policy}',kind='evaluate',config=asdict(config),weights_from=None))
    return learning,baselines+evaluation


def completed_rallies(rows):
    result=[];hits=0;seconds=0.
    for row in rows:
        hits+=row['hits'];seconds+=row['elapsed_seconds']
        if row['terminated'] or row['truncated']:
            result.append(dict(reward=row['reward'],hits=hits,seconds=seconds,
                               terminated=row['terminated'],truncated=row['truncated']))
            hits=0;seconds=0.
    return result


def checkpoint_open_run(run):
    """Save the full chunk boundary without closing a healthy live writer.

    Actual opens still execute DurableRun's complete recovery verification. A
    failed checkpoint poisons this writer exactly as stop/step failures do.
    """
    if run.failed or run.stopped:
        raise RuntimeError('writer unavailable; reopen journal')
    run.failed=True
    run.db.execute('BEGIN IMMEDIATE')
    try:
        count=run.db.execute('SELECT count(*) FROM replay_samples').fetchone()[0]
        if count!=run.session.transitions:
            raise ValueError('stale chunk checkpoint')
        run._checkpoint()
        run.db.execute('COMMIT')
    except BaseException:
        run.db.execute('ROLLBACK')
        raise
    run.last_snapshot_count=run.session.transitions
    run.last_snapshot_mono=run.clock.monotonic_ns()
    run._timing('chunk_checkpoint')
    run.failed=False


def validate_first_hit_job(job):
    mode=job.get('first_hit_cutoff',False)
    if (type(mode) is not bool or (mode and
            (job['kind']!='evaluate' or job.get('purpose')!='training_fitness' or job['config'].get('episodes')!=1))):
        raise ValueError('first-hit cutoff is single-episode training fitness only')
    return mode


def validate_camera_job(job):
    mode=validate_camera_intervention(job.get('camera_intervention'))
    config=job.get('config',{})
    if mode is not None and (job['kind']!='evaluate' or job.get('purpose')!='fresh_evaluation'
            or config.get('episodes')!=1 or config.get('policy_kind')!='trained'
            or config.get('blank_camera',False) or config.get('skip_baseline_perception',False)
            or job.get('first_hit_cutoff',False)):
        raise ValueError('camera intervention is single-episode frozen evaluation only')
    return mode


def validate_actor_job(job):
    mode=job.get('actor')
    if mode is not None:
        if (type(mode) is not dict or set(mode) not in ({'theta_from','temperature','action_seed'},
                {'theta_from','temperature','action_seed','noise'})
                or not validate_first_hit_job(job) or job.get('camera_intervention') is not None
                or job.get('projection_from') is None or job.get('scaling_from') is not None
                or job['config'].get('action_hold_frames',1)!=1):
            raise ValueError('actor is projected K=1 first-hit training only')
        validate_bias(mode.get('noise'))
    return mode


def worker(directory,job_id,max_work_seconds=160.,*,emit=True,_live_run=None,_keep_open=False,verify_only=False):
    if not 0 < max_work_seconds <= 160.:
        raise ValueError('worker work budget must be in (0,160] seconds')
    torch.set_num_threads(1)
    started=time.monotonic()
    manifest=json.loads((directory/'manifest.json').read_text())
    if (manifest['source'] != source_signature() or manifest['runtime'] != runtime_signature()
            or manifest['driver_hash'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError('source/runtime differs from declared experiment')
    job=next(row for row in manifest['jobs'] if row['id']==job_id)
    if job['kind']!='learn' and job.get('reward_shaping') is not None:
        raise ValueError('diagnostic reward transforms are training-only; evaluation is raw score-only')
    if job['kind']!='learn' and job.get('exploration_block_frames') is not None:
        raise ValueError('exploration blocks are training-only; fresh evaluation is frozen greedy')
    first_hit_cutoff=validate_first_hit_job(job)
    camera_intervention=validate_camera_job(job)
    actor=validate_actor_job(job)
    checkpoints=job.get('checkpoints',(2000,6000))
    scaling=None if job.get('scaling_from') is None else json.loads((directory/job['scaling_from']).read_text())
    projection=None if job.get('projection_from') is None else json.loads((directory/job['projection_from']).read_text())
    if projection is not None and ((job['kind']!='learn' and actor is None) or scaling is not None):
        raise ValueError('projection is training-only and owns its cap; evaluate mapped weights with their exported cap')
    evaluation_config=None if job['kind']=='learn' else dict(job['config'])
    if evaluation_config is not None and scaling is not None:evaluation_config['policy_scaling']=scaling
    db=directory/(job_id+'.db')
    if _live_run is not None:
        run=_live_run
        actual=Path(run.db.execute('PRAGMA database_list').fetchone()[2]).resolve()
        if actual!=db.resolve() or run.failed or run.stopped:
            raise ValueError('live writer differs from requested healthy journal')
    elif db.exists():
        run=DurableRun(db)
    else:
        seed=job.get('seed',DEV_SEED)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if job['kind']=='learn':
            session=LearningSession(job['seed'],max_episode_ticks=2400,lam=job['lam'],
                                    alpha=job.get('alpha',.0001),policy_scaling=scaling,
                                    opponent_speed=job.get('opponent_speed'),action_hold_frames=job.get('action_hold_frames',1),
                                    policy_projection=projection,reward_shaping=job.get('reward_shaping'),
                                    exploration_block_frames=job.get('exploration_block_frames'))
        else:
            weights=None if job['weights_from'] is None else np.load(directory/job['weights_from'],allow_pickle=False)
            if weights is not None:
                metadata=json.loads((directory/job['weights_from']).with_suffix('.json').read_text())
                if list(weights.shape)!=metadata['shape'] or hashlib.sha256(weights.tobytes()).hexdigest()!=metadata['weight_hash']:
                    raise ValueError('evaluation weights differ from checkpoint metadata')
            options={}
            if first_hit_cutoff:options['first_hit_cutoff']=True
            if camera_intervention is not None:options['camera_intervention']=camera_intervention
            if actor is None:
                session=FreshPongEvaluator(weights,PongEvaluationConfig(**evaluation_config),**options)
            else:
                theta=np.load(directory/actor['theta_from'],allow_pickle=False)
                session=FirstHitActorEpisode(theta,PongEvaluationConfig(**evaluation_config),projection=projection,
                    temperature=actor['temperature'],action_seed=actor['action_seed'],noise=actor.get('noise'))
                if not np.array_equal(weights,session.policy.weights):raise ValueError('actor mapped input differs')
        run=DurableRun(db,session=session)
    session=run.session;before=session.transitions
    if job['kind']=='learn':
        expected_config=dict(seed=job['seed'],max_episode_ticks=2400,alpha=job.get('alpha',.0001),lam=job['lam'],
                             gamma_per_second=.95,epsilon=.2)
        if scaling is not None:expected_config['policy_scaling']=scaling
        if projection is not None:expected_config['policy_projection']=projection
        if job.get('reward_shaping') is not None:expected_config['reward_shaping']=job['reward_shaping']
        if 'opponent_speed' in job:expected_config['opponent_speed']=job['opponent_speed']
        if job.get('exploration_block_frames') is not None:expected_config['exploration_block_frames']=job['exploration_block_frames']
        hold=job.get('action_hold_frames',1)
        if hold!=1:expected_config.update(action_hold_frames=hold,macro_lambda=job['lam']**hold)
        if type(session) is not LearningSession or session.config != expected_config:
            raise ValueError('training journal differs from declared configuration')
        export_training_checkpoint(directory,job_id,session,checkpoints=checkpoints)
    elif type(session) is not (FreshPongEvaluator if actor is None else FirstHitActorEpisode) or asdict(session.config) != evaluation_config:
        raise ValueError('evaluation journal differs from declared configuration')
    if actor is not None and (session.temperature!=actor['temperature'] or session.actor_action_seed!=actor['action_seed']
            or getattr(session,'actor_bias_config',None)!=actor.get('noise')
            or session.actor_projection.config!=projection or not np.array_equal(session.theta,np.load(directory/actor['theta_from'],allow_pickle=False))):
        raise ValueError('actor journal differs from declared head/exploration')
    if job['kind']!='learn' and getattr(session,'first_hit_cutoff',False)!=first_hit_cutoff:
        raise ValueError('restored task boundary differs from declared training purpose')
    if job['kind']!='learn' and getattr(session,'camera_intervention',None)!=camera_intervention:
        raise ValueError('restored camera intervention differs from declared evaluation')
    replayed=run.replayed if _live_run is None else 0;latencies=[]
    def complete():
        return session.transitions==job['transitions'] if job['kind'] in ('learn','calibration') else session.complete
    if verify_only and not complete():
        run.close_without_checkpoint()
        raise ValueError('verification-only recovery requires an actually complete durable session')
    while not complete() and session.transitions-before<200 and time.monotonic()-started<max_work_seconds:
        step_started=time.monotonic();record=run.step();latencies.append(time.monotonic()-step_started)
        if job['kind']=='learn' and job.get('max_abs_q') is not None:
            magnitude=max(abs(value) for value in (record['q'],record['q_next'],record['q_spread']) if value is not None)
            if magnitude>job['max_abs_q']:
                failure=dict(job=job_id,transition=record['transition'],magnitude=magnitude,
                    threshold=job['max_abs_q'],reason='prespecified numerical resource stop',qualifying_endpoint=False)
                run.stop()
                atomic_json(directory/(job_id+'-numerical-stop.json'),failure)
                raise FloatingPointError('prespecified numerical resource stop')
        if job['kind']=='learn' and session.transitions in checkpoints:
            export_training_checkpoint(directory,job_id,session,checkpoints=checkpoints)
    rows=[json.loads(row[0]) for row in run.db.execute('SELECT record FROM replay_samples ORDER BY seq')]
    report=dict(job=job_id,kind=job['kind'],complete=complete(),transitions=session.transitions,
        new_transitions=session.transitions-before,replayed_transitions=replayed,
        episodes=session.episodes,initial_cameras=session.initial_cameras,
        transition_cameras=session.transition_cameras,physics_seconds=session.physics_seconds,
        chunk_p50_seconds=float(np.median(latencies)) if latencies else None,
        chunk_p95_seconds=float(np.quantile(latencies,.95)) if latencies else None)
    if job['kind']!='learn':
        report.update(session.summary())
        if actor is not None and session.complete:
            path=directory/(job_id+'-eligibility.npy')
            if path.exists() and not np.array_equal(np.load(path,allow_pickle=False),session.eligibility):
                raise ValueError('completed actor eligibility changed')
            if not path.exists():atomic_weights(path,session.eligibility)
        if job['kind']=='calibration':
            report['environment_episode_schedule_complete']=report['complete']
            report['complete']=complete()
    else:
        scoring=[row for row in rows if row['terminated'] or row['truncated']]
        rallies=completed_rallies(rows)
        report.update(total_reward=sum(row['reward'] for row in rows),
            wins=sum(row['reward']>0 for row in scoring),losses=sum(row['reward']<0 for row in scoring),
            hits=sum(row['hits'] for row in rows),truncations=sum(row['truncated'] for row in rows),
            action_counts={str(a):sum(row['action']==a for row in rows) for a in range(3)},
            weight_norm=float(np.linalg.norm(session.learner.weights)),
            feature_norm2_quantiles=np.quantile([row['feature_norm2'] for row in rows],[0,.5,1]).tolist() if rows else [],
            q_spread_quantiles=np.quantile([row['q_spread'] for row in rows],[0,.5,1]).tolist() if rows else [])
        report.update(rallies=rallies,opponent_speed=session.env.game.config.opponent_speed,
            natural_serve_wins=sum(r['reward']>0 and r['hits']==0 for r in rallies),
            own_hit_followed_wins=sum(r['reward']>0 and r['hits']>0 for r in rallies))
        report['finite_weights']=bool(np.isfinite(session.learner.weights).all())
        report['max_absolute_weight']=float(np.max(np.abs(session.learner.weights)))
        report['max_policy_feature_norm2']=max((row['policy_feature_norm2'] for row in rows),default=0.)
        if job.get('exploration_block_frames') is not None:
            selected=[r['exploration'] for r in rows]
            exploratory=sum(r['exploratory'] for r in selected)
            nongreedy=sum(r['non_greedy'] for r in selected)
            report.update(exploration_block_frames=job['exploration_block_frames'],
                exploratory_transitions=exploratory,non_greedy_transitions=nongreedy,
                exploratory_fraction=exploratory/len(rows) if rows else None,
                non_greedy_fraction=nongreedy/len(rows) if rows else None,
                exploratory_physical_seconds=sum(r['elapsed_seconds'] for r in rows if r['exploration']['exploratory']),
                non_greedy_physical_seconds=sum(r['elapsed_seconds'] for r in rows if r['exploration']['non_greedy']),
                exploration_blocks_started=sum(r['block_frame']==1 for r in selected),
                exploratory_blocks_started=sum(r['block_frame']==1 and r['exploratory'] for r in selected),
                learner_updates=session.learner.transitions)
        if job.get('reward_shaping') is not None:
            report.update(reward_shaping=session.reward_shaping,
                total_learning_reward=sum(row['learning_reward'] for row in rows),
                total_hit_reward=sum(row['hit_reward'] for row in rows),
                total_survival_reward=sum(row['survival_reward'] for row in rows),
                raw_score_reward=sum(row['reward'] for row in rows))
            if 'score_coefficient' in session.reward_shaping:
                report['total_score_reward']=sum(row['score_reward'] for row in rows)
        if getattr(session,'action_hold_frames',1)!=1:
            report.update(action_hold_frames=session.action_hold_frames,macro_choices=session.macro_choices,
                          macro_updates=session.learner.transitions,macro_lambda=session.learner.lam,
                          pending_hold_frames=session.hold_frames)
    if _keep_open:checkpoint_open_run(run)
    else:run.stop()
    # Read the persisted save event, so chunk saving cost is included in prefixes.
    with sqlite3.connect(db) as connection:
        if job.get('compact_reports',False):
            report['timing_event_count']=connection.execute('SELECT count(*) FROM replay_timing').fetchone()[0]
            report['last_timing_event']=connection.execute('SELECT kind,count,process_ns,gap_ns FROM replay_timing ORDER BY id DESC LIMIT 1').fetchone()
        else:
            report['timing_events']=connection.execute('SELECT kind,count,process_ns,gap_ns FROM replay_timing ORDER BY id').fetchall()
    report['worker_wall_seconds']=time.monotonic()-started
    atomic_json(directory/(job_id+'.json'),report)
    if emit:print(json.dumps({k:report[k] for k in ('job','complete','transitions','replayed_transitions','worker_wall_seconds')}),flush=True)
    return (report,run) if _keep_open else report


def multiple_chunks(directory,job_id,max_work_seconds=160.,max_chunks=5,*,warm_session=False):
    """Reuse imports and optionally the writer; every chunk has a full snapshot."""
    if type(max_chunks) is not int or not 1<=max_chunks<=5:raise ValueError('one to five bounded chunks required')
    if not 0<max_work_seconds<=160.:raise ValueError('worker work budget must be in (0,160] seconds')
    started=time.monotonic();chunks=[];live=None
    try:
        for _ in range(max_chunks):
            remaining=max_work_seconds-(time.monotonic()-started)
            if remaining<=20:break
            if warm_session:
                report,live=worker(directory,job_id,remaining,emit=False,_live_run=live,_keep_open=True)
            else:report=worker(directory,job_id,remaining,emit=False)
            chunks.append({key:report[key] for key in ('transitions','new_transitions','replayed_transitions','worker_wall_seconds')})
            if report['complete']:break
        if live is not None:
            save_started=time.monotonic()
            live.stop()
            db=directory/(job_id+'.db')
            with sqlite3.connect(db) as connection:
                if 'timing_event_count' in report:
                    report['timing_event_count']=connection.execute('SELECT count(*) FROM replay_timing').fetchone()[0]
                    report['last_timing_event']=connection.execute('SELECT kind,count,process_ns,gap_ns FROM replay_timing ORDER BY id DESC LIMIT 1').fetchone()
                else:report['timing_events']=connection.execute('SELECT kind,count,process_ns,gap_ns FROM replay_timing ORDER BY id').fetchall()
            report['final_stop_seconds']=time.monotonic()-save_started
            report['worker_wall_seconds']+=report['final_stop_seconds']
            chunks[-1]['worker_wall_seconds']=report['worker_wall_seconds']
            atomic_json(directory/(job_id+'.json'),report)
    except BaseException:
        if live is not None and not live.stopped:live.close_without_checkpoint()
        raise
    if not chunks:raise ValueError('insufficient work budget for a checkpointed chunk')
    result=dict(job=job_id,complete=report['complete'],transitions=report['transitions'],
        new_transitions=sum(c['new_transitions'] for c in chunks),
        replayed_transitions=sum(c['replayed_transitions'] for c in chunks),
        worker_wall_seconds=time.monotonic()-started,chunks=chunks,warm_session=warm_session)
    print(json.dumps(result),flush=True)
    return result


def supervise(directory,workers=3,chunks_per_worker=5):
    directory.mkdir(parents=True,exist_ok=True)
    manifest_path=directory/'manifest.json';learning,evaluation=jobs();expected=learning+evaluation
    driver_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if (manifest['source']!=source_signature() or manifest['runtime']!=runtime_signature()
                or manifest['driver_hash']!=driver_hash or manifest['jobs']!=expected):
            raise ValueError('changed experiment source/runtime/driver/allocation')
    else:
        manifest=dict(protocol='docs/reconstruction/2026-10-15-pong-development-protocol.md',
            started_wall_ns=time.time_ns(),source=source_signature(),runtime=runtime_signature(),
            model_hash=MODEL_HASH,graph_digest=GRAPH_DIGEST,driver_hash=driver_hash,jobs=expected,
            long_bank_entries=0,context_fallback_steps=4,torch_threads=1,workers=workers,
            learning_transition_cap=36000,evaluation_transition_cap=216000,
            aggregate_wall_seconds=5400,reserved_final_seeds=list(FINAL_SEEDS),
            chunks_per_worker=chunks_per_worker,
            dependencies=sorted([(d.metadata['Name'],d.version) for d in distributions()]))
        atomic_json(manifest_path,manifest)
    def run_job(job):
        report_path=directory/(job['id']+'.json')
        while True:
            if report_path.exists() and json.loads(report_path.read_text())['complete']:
                return dict(job=job['id'],status='complete')
            remaining=5400-(time.time_ns()-manifest['started_wall_ns'])/1e9
            if remaining<25:
                return dict(job=job['id'],status='partial',failure='aggregate budget exhausted')
            command=[sys.executable,__file__,'--directory',str(directory),'--worker',job['id'],
                     '--max-work-seconds',str(min(160.,remaining-25)),
                     '--chunks-per-worker',str(chunks_per_worker)]
            started=time.monotonic()
            try:
                result=subprocess.run(command,capture_output=True,text=True,timeout=min(180.,remaining-5))
                cost=dict(job=job['id'],wall_seconds=time.monotonic()-started,returncode=result.returncode,
                          stdout=result.stdout[-1500:],stderr=result.stderr[-4000:])
            except subprocess.TimeoutExpired:
                cost=dict(job=job['id'],wall_seconds=time.monotonic()-started,returncode=None,timeout=True)
            # One independent cost file per job avoids interleaved writes.
            with (directory/(job['id']+'-costs.jsonl')).open('a') as out:
                out.write(json.dumps(cost)+'\n');out.flush();os.fsync(out.fileno())
            print(json.dumps(cost),flush=True)
            if cost['returncode']:
                return dict(job=job['id'],status='failed',failure=cost['stderr'])
            if cost['returncode'] is None:
                return dict(job=job['id'],status='partial',failure='worker timeout; no blind retries')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        training_results=list(pool.map(run_job,learning))
    results=training_results
    if all(row['status']=='complete' for row in training_results):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results+=list(pool.map(run_job,evaluation))
    reports=[]
    for job in expected:
        path=directory/(job['id']+'.json')
        if path.exists():reports.append(json.loads(path.read_text()))
    counts={'learn':0,'evaluate':0}
    for job in expected:
        path=directory/(job['id']+'.db')
        if path.exists():
            with sqlite3.connect(path) as connection:
                counts[job['kind']]+=connection.execute('SELECT count(*) FROM replay_samples').fetchone()[0]
    assert counts['learn']<=36000 and counts['evaluate']<=216000
    summary=dict(status='complete' if len(results)==len(expected) and all(r['status']=='complete' for r in results) else 'partial',
        wall_seconds=(time.time_ns()-manifest['started_wall_ns'])/1e9,committed_counts=counts,
        results=results,reports=reports,reserved_final_seeds_unused=list(FINAL_SEEDS))
    atomic_json(directory/'summary.json',summary)
    print(json.dumps(dict(status=summary['status'],counts=counts,wall_seconds=summary['wall_seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--worker')
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--max-work-seconds',type=float,default=160.)
    parser.add_argument('--chunks-per-worker',type=int,choices=(1,2,3,4,5),default=5)
    parser.add_argument('--warm-session',action='store_true',help='retain one verified writer across bounded chunks')
    args=parser.parse_args()
    if args.worker:multiple_chunks(args.directory,args.worker,args.max_work_seconds,args.chunks_per_worker,warm_session=args.warm_session)
    else:supervise(args.directory,args.workers,args.chunks_per_worker)
