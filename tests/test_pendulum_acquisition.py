"""Narrow fixed-dose, ownership, scheduling and frozen-analysis contracts."""
import copy,json,sys,time
from concurrent.futures import Future
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import run_pendulum_acquisition as study
import pendulum_acquisition_worker as worker
import pendulum_acquisition_metrics as metrics
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.durable import DurableRun,runtime_signature
from fly_connectome.control.online_vision import VisionSnapshot
from fly_connectome.control.projection import PROJECTION_VERSION
from fly_connectome.control.session import source_signature,MODEL_HASH,VISION_CONFIGURATION
from fly_connectome.vision import load_default,VisualStateEncoder


def test_seed_and_training_contract_is_independent_and_deterministic(monkeypatch):
    monkeypatch.setattr(study,'ROOTS',dict(environment=104729,learner=104743,development=104759,control_actions=104761))
    monkeypatch.setattr(study,'TAG',104773)
    schedule=study.schedule([])
    assert len(set(schedule['development']))==64 and schedule==study.schedule([])
    assert not set(schedule['development'])&set(study.FINAL_SEEDS)
    assert not set(study.schedule(schedule['development'][:3])['development'])&set(schedule['development'][:3])
    jobs=study.training_jobs(dict(starts={a:dict(file=a+'.pkl') for a in study.ARMS},
        fixed_artifacts={a+'.pkl':'a'*64 for a in study.ARMS}|{'search-transform.npy':'b'*64}))
    assert all(j['config']['seed']!=j['config']['learner_seed'] for j in jobs)
    assert jobs[0]['config']==jobs[1]['config'] and jobs[0]['checkpoints']==[5000,10000,20000]


def test_paired_metrics_require_order_and_recompute_max_control():
    groups={name:[dict(seed=i+100,serve_index=i,survival_seconds=4. if name.endswith('endpoint') else 1.,
        time_limit=(i<40 if name.endswith('endpoint') else False)) for i in range(64)]
        for name in ('experienced_endpoint','fresh_endpoint',*metrics.CONTROLS)}
    result=metrics.survival_evidence(groups,1,draws=40)
    assert result['arms']['experienced_endpoint']['competence_screen']
    assert result['arms']['experienced_endpoint']['ci95']==[3.,3.]
    assert result['experienced_minus_fresh']['point']==0.
    groups['noop'].reverse()
    with pytest.raises(ValueError,match='cohort ordering'):metrics.survival_evidence(groups,1,draws=40)


def test_failed_cost_and_original_phase_clock_survive_restart(tmp_path,monkeypatch):
    class Pool:
        def __init__(self,*a,**k):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def submit(self,*a):
            f=Future();f.set_result(dict(ok=False,job='x',wall_seconds=1.));return f
    monkeypatch.setattr(study,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(study.worker,'complete_report',lambda *a:None)
    m=dict(started_wall_ns=time.time_ns());jobs=[dict(id='x')]
    b=study.execute_phase(tmp_path,m,'training',jobs,2)
    assert b['failed'] and not b['complete']
    assert study.execute_phase(tmp_path,m,'training',jobs,2)==b
    study.atomic_json(tmp_path/'training-boundary.json',dict(b,failed=False))
    with pytest.raises(ValueError,match='completion changed'):study.execute_phase(tmp_path,m,'training',jobs,2)


def test_retained_summary_does_not_rescore_against_later_clock(tmp_path,monkeypatch):
    evidence=dict(fixed='evidence');report=dict(status='complete',wall_seconds=2000)
    monkeypatch.setattr(study,'outcome_evidence',lambda *a:evidence)
    study.atomic_json(tmp_path/'study-completion.json',dict(summary=report,evidence=evidence))
    assert study.retained_outcome(tmp_path,{})==report
    study.atomic_json(tmp_path/'summary.json',dict(report,wall_seconds=9999))
    with pytest.raises(ValueError,match='summary changed'):study.retained_outcome(tmp_path,{})


def make_training_fixture(directory):
    directory.mkdir();vision=load_default();vision.reset_state()
    snapshot=VisionSnapshot(vision,fingerprint(vision),source_signature(),MODEL_HASH,VISION_CONFIGURATION,0)
    worker.save_snapshot(directory/'before.pkl',snapshot)
    dimension=VisualStateEncoder(64,64,.02).dimension
    matrix=np.zeros((4,dimension));matrix[:,[0,1,2,dimension-1]]=np.eye(4)
    worker.save_array(directory/'search-transform.npy',matrix)
    projection=dict(version=PROJECTION_VERSION,matrix_file=str(directory/'search-transform.npy'),
        file_sha256=worker.sha(directory/'search-transform.npy'),array_sha256=__import__('hashlib').sha256(matrix.tobytes()).hexdigest(),
        shape=list(matrix.shape),max_norm2=256.,calibration=dict(test_only=True))
    job=dict(id='trial',arm='test',kind='train',phase='training',camera_dose=12,checkpoints=[5,8,12],
        snapshot_from='before.pkl',input_hashes={n:worker.sha(directory/n) for n in ('before.pkl','search-transform.npy')},
        config=dict(seed=739397,learner_seed=739409,max_episode_ticks=7,alpha=.001,lam=.995,epsilon=.2,gamma_per_second=.95))
    m=dict(source=source_signature(),runtime=runtime_signature(),pinned_sources={},projection=projection,jobs=[job])
    study.atomic_json(directory/'manifest.json',m);study.atomic_json(directory/'trial-spec.json',job)
    deadline=time.time()+60;study.atomic_json(directory/'training-boundary.json',dict(deadline_wall_ns=int(deadline*1e9)))
    return job,int(deadline*1e9)/1e9


def test_worker_recovers_boundary_export_and_freezes_real_handoff(tmp_path,monkeypatch):
    torch.set_num_threads(1);directory=tmp_path/'writer';job,deadline=make_training_fixture(directory)
    original=worker.export_boundary
    def fail_at_first(*a):raise RuntimeError('simulated crash after durable checkpoint before export')
    monkeypatch.setattr(worker,'export_boundary',fail_at_first)
    failed=worker.execute_job(directory,job,deadline)
    assert not failed['ok'] and failed['wall_seconds']>0
    restored=DurableRun(directory/'trial.db')
    assert restored.session.transitions==5 and restored.replayed==0
    restored.close_without_checkpoint()
    monkeypatch.setattr(worker,'export_boundary',original)
    cost=worker.execute_job(directory,job,deadline)
    assert cost['ok'] and cost['complete'] and cost['new_camera_transitions']==7
    report=worker.complete_report(directory,job)
    assert set(report['exports'])=={'5','8','12'}
    assert report['session']['visual_observations']==report['session']['transitions']+report['session']['initial_cameras']
    assert report['session']['cumulative_visual_learning_observations']==report['session']['visual_observations']
    # Real exported mapped policy and retained memory, consumed by a frozen B probe.
    prefix='test_at12';config=study.PendulumEvaluationConfig(739411,max_episode_ticks=7,
        policy_scaling=json.loads((directory/(prefix+'-cap.json')).read_text()))
    from fly_connectome.control.pendulum_evaluation import FreshPendulumEvaluator
    snapshot=worker.load_snapshot(directory,prefix+'-vision.pkl')
    probe=FreshPendulumEvaluator(np.load(directory/(prefix+'-weights.npy')),config,vision_snapshot=snapshot)
    learned=probe.learned_hash()
    while not probe.complete:probe.step()
    assert probe.learned_hash()==learned and probe.summary()['visual_learning_observations']==0
    # Actual zero-new-camera completion recovery, then stale-report rejection.
    cost=worker.execute_job(directory,job,deadline)
    assert cost['new_camera_transitions']==cost['replayed_camera_transitions']==0
    path=directory/'trial.json';bad=json.loads(path.read_text());bad['session']['transitions']-=1
    study.atomic_json(path,bad)
    with pytest.raises(ValueError,match='journal rows'):worker.complete_report(directory,job)
