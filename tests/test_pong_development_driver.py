import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from fly_connectome.control.projection import PROJECTION_VERSION,FixedPolicyProjection
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))


spec=importlib.util.spec_from_file_location('pong_driver',Path(__file__).parents[1]/'scripts/experimental/run_pong_development.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def test_declared_allocation_does_not_use_final_holdout():
    learning,evaluation=driver.jobs()
    assert len(learning)==6 and sum(job['transitions'] for job in learning)==36000
    assert len(evaluation)==15 and sum(job['config']['episodes'] for job in evaluation)==180
    assert {job['config']['seed'] for job in evaluation} == {driver.DEV_SEED}
    assert not set(driver.TRAIN_SEEDS).intersection(driver.FINAL_SEEDS)


def test_weight_export_recovers_boundary_crash_and_rejects_later_substitution(tmp_path):
    weights=np.array([[.1,.2]])
    driver.export_checkpoint(tmp_path,'run',2000,weights)
    np.testing.assert_array_equal(np.load(tmp_path/'run_at2000.npy'),weights)
    driver.export_checkpoint(tmp_path,'run',2000,weights)
    with pytest.raises(ValueError,match='disagree'):
        driver.export_checkpoint(tmp_path,'run',2000,weights+1)
    (tmp_path/'run_at2000.npy').unlink()
    with pytest.raises(ValueError,match='missing past checkpoint'):
        driver.export_checkpoint(tmp_path,'run',2001,weights)
    assert not (tmp_path/'run_at2000.npy').exists()


def test_declared_longer_checkpoints_recover_without_substituting_later_weights(tmp_path):
    weights=np.array([[.3,.4]])
    driver.export_checkpoint(tmp_path,'long',15000,weights,checkpoints=(15000,30000))
    np.testing.assert_array_equal(np.load(tmp_path/'long_at15000.npy'),weights)
    driver.export_checkpoint(tmp_path,'long',30000,weights+1,checkpoints=(15000,30000))
    (tmp_path/'long_at15000.npy').unlink()
    with pytest.raises(ValueError,match='missing past checkpoint'):
        driver.export_checkpoint(tmp_path,'long',30000,weights+1,checkpoints=(15000,30000))


def projected_session(tmp_path,count):
    matrix=np.array([[1.,2.,0.],[-1.,0.,3.]])
    path=tmp_path/'matrix.npy';np.save(path,matrix,allow_pickle=False)
    config=dict(version=PROJECTION_VERSION,matrix_file=str(path),shape=list(matrix.shape),
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),array_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
        max_norm2=256.,calibration={})
    projection=FixedPolicyProjection(3,config)
    session=SimpleNamespace(transitions=count,projection=projection,
        learner=SimpleNamespace(weights=np.arange(6,dtype=np.float64).reshape(3,2)))
    session.mapped_evaluation=lambda:projection.mapped_evaluation(session.learner.weights)
    return session


def test_projected_checkpoint_boundary_recovers_cap_and_theta_exports(tmp_path):
    session=projected_session(tmp_path,2)
    driver.export_training_checkpoint(tmp_path,'p',session,checkpoints=(2,4))
    mapped,cap=session.mapped_evaluation()
    np.testing.assert_array_equal(np.load(tmp_path/'p_at2.npy'),mapped)
    np.testing.assert_array_equal(np.load(tmp_path/'p_at2.theta.npy'),session.learner.weights)
    assert json.loads((tmp_path/'p_at2.cap.json').read_text())==cap
    (tmp_path/'p_at2.theta.npy').unlink();(tmp_path/'p_at2.cap.json').unlink()
    driver.export_training_checkpoint(tmp_path,'p',session,checkpoints=(2,4))
    (tmp_path/'p_at2.cap.json').unlink()
    driver.export_training_checkpoint(tmp_path,'p',session,checkpoints=(2,4))
    session.transitions=3
    driver.export_training_checkpoint(tmp_path,'p',session,checkpoints=(2,4))
    (tmp_path/'p_at2.cap.json').unlink()
    with pytest.raises(ValueError,match='missing past projected checkpoint'):
        driver.export_training_checkpoint(tmp_path,'p',session,checkpoints=(2,4))


def test_past_projected_checkpoint_theta_cannot_be_replaced(tmp_path):
    session=projected_session(tmp_path,2)
    driver.export_training_checkpoint(tmp_path,'p',session,checkpoints=(2,4))
    theta=np.load(tmp_path/'p_at2.theta.npy');theta[0,0]+=1
    np.save(tmp_path/'p_at2.theta.npy',theta,allow_pickle=False)
    session.transitions=3
    with pytest.raises(ValueError,match='provenance changed'):
        driver.export_training_checkpoint(tmp_path,'p',session,checkpoints=(2,4))


def test_projected_worker_exports_to_independent_mapped_evaluator(tmp_path):
    matrix=np.zeros((4,4389),dtype=np.float64)
    matrix[0,4384]=1.;matrix[1,4388]=1.;matrix[2,4253]=1.;matrix[3,0]=1.
    path=tmp_path/'matrix.npy';np.save(path,matrix,allow_pickle=False)
    config=dict(version=PROJECTION_VERSION,matrix_file=str(path),shape=list(matrix.shape),
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),array_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
        max_norm2=256.,calibration={})
    driver.atomic_json(tmp_path/'projection.json',config)
    learn=dict(id='projected',kind='learn',seed=782317,lam=.995,alpha=.001,
        opponent_speed=.25,transitions=4,checkpoints=[4],projection_from='projection.json')
    evaluation=dict(id='mapped_eval',kind='evaluate',weights_from='projected_at4.npy',
        scaling_from='projected_at4.cap.json',config=driver.asdict(driver.PongEvaluationConfig(
            927317,episodes=1,max_episode_ticks=5,opponent_speed=.25)))
    driver.atomic_json(tmp_path/'manifest.json',dict(source=driver.source_signature(),
        runtime=driver.runtime_signature(),driver_hash=hashlib.sha256(Path(driver.__file__).read_bytes()).hexdigest(),
        jobs=[learn,evaluation]))
    report=driver.worker(tmp_path,'projected',emit=False)
    assert report['complete'] and report['transitions']==4
    run=driver.DurableRun(tmp_path/'projected.db')
    weights,cap=run.session.mapped_evaluation();theta=run.session.learner.weights.copy()
    run.close_without_checkpoint()
    from run_dct_sarsa_pong import pin_completion,complete_report
    pin_completion(tmp_path,learn)
    assert complete_report(tmp_path,learn)['transitions']==4
    (tmp_path/'projected-completion.json').unlink()
    verified=driver.worker(tmp_path,'projected',emit=False)
    assert verified['complete'] and verified['new_transitions']==0
    pin_completion(tmp_path,learn)
    assert complete_report(tmp_path,learn)['transitions']==4
    report=driver.worker(tmp_path,'mapped_eval',emit=False)
    assert report['complete'] and report['episodes']==1
    run=driver.DurableRun(tmp_path/'mapped_eval.db')
    np.testing.assert_array_equal(run.session.policy.weights,weights)
    assert run.session.config.policy_scaling==cap
    assert not hasattr(run.session,'projection')
    assert np.load(tmp_path/'projected_at4.theta.npy').shape==theta.shape==(3,4)
    run.close_without_checkpoint()
