"""Explicit research integrity contract; reference behavior remains the default."""
import copy
import hashlib
import json
import math
import numpy as np
import pytest
import torch
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.durable import DurableRun,RESEARCH_UNCHECKED,RESEARCH_EXECUTION
from fly_connectome.control.pong_actor import FirstHitActorEpisode
from fly_connectome.control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig
from fly_connectome.control.projection import PROJECTION_VERSION


def session(tmp_path,kind='ar'):
    torch.set_num_threads(1)
    config=PongEvaluationConfig(710191,episodes=1,max_episode_ticks=80,opponent_speed=.65)
    if kind=='greedy':return FreshPongEvaluator(np.zeros((3,4389)),config)
    matrix=np.zeros((2,4389));matrix[0,0]=100.;matrix[1,1]=50.
    path=tmp_path/'matrix.npy';np.save(path,matrix)
    projection=dict(version=PROJECTION_VERSION,matrix_file=str(path),shape=[2,4389],
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),array_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
        max_norm2=256.,calibration={})
    noise=None if kind=='plain' else dict(kind='centered-gaussian-ar1-v1',sigma=.06,
        rho=0. if kind=='iid' else math.exp(-.02/.25),seed=827193)
    return FirstHitActorEpisode(np.array([[.01,-.02],[.02,.01],[-.01,.01]]),config,
        projection=projection,temperature=.1,action_seed=598933,noise=noise)


@pytest.mark.parametrize('kind',['plain','iid','ar','greedy'])
def test_research_exact_records_terminal_graph_and_unchecked_tail_recovery(tmp_path,kind):
    initial=session(tmp_path,kind)
    reference=DurableRun(tmp_path/'reference.db',session=copy.deepcopy(initial))
    fast=DurableRun(tmp_path/'research.db',session=copy.deepcopy(initial),research_check_interval=16)
    for _ in range(3):assert reference.step()==fast.step()
    assert fast.db.execute('SELECT state_hash FROM replay_samples').fetchall()==[(RESEARCH_UNCHECKED,)]*3
    expected=fingerprint(fast.session);fast.close_without_checkpoint()
    fast=DurableRun(tmp_path/'research.db',research_check_interval=16)
    assert fast.replayed==3 and fast.last_research_full_check['count']==3
    assert fingerprint(fast.session)==expected==fingerprint(reference.session)
    while not reference.session.complete:
        assert reference.step()==fast.step()
    assert fingerprint(reference.session)==fingerprint(fast.session)
    assert fast.last_research_full_check['count']==fast.session.transitions
    assert fast.db.execute('SELECT state_hash FROM replay_samples ORDER BY seq DESC LIMIT 1').fetchone()[0]!=RESEARCH_UNCHECKED
    reference.stop();fast.stop()


def test_nonboundary_checkpoint_and_recovery_exit_full_guard(tmp_path):
    path=tmp_path/'run.db'
    run=DurableRun(path,session=session(tmp_path),snapshot_transitions=5,research_check_interval=16)
    for _ in range(6):run.step()
    assert run.db.execute('SELECT count FROM replay_active').fetchone()[0]==5
    hashes=run.db.execute('SELECT state_hash FROM replay_samples ORDER BY seq').fetchall()
    assert hashes[4][0]!=RESEARCH_UNCHECKED and hashes[5][0]==RESEARCH_UNCHECKED
    expected=fingerprint(run.session);run.close_without_checkpoint()
    run=DurableRun(path,research_check_interval=16)
    assert run.replayed==1 and run.last_research_full_check['count']==6
    assert fingerprint(run.session)==expected
    run.stop()
    run=DurableRun(path,research_check_interval=16)
    assert run.replayed==0 and run.last_research_full_check['count']==6
    run.stop()


def test_persistent_parameter_mutation_fails_before_boundary_commit(tmp_path):
    run=DurableRun(tmp_path/'run.db',session=session(tmp_path),research_check_interval=16)
    run.session.theta[0,0]+=.01
    for _ in range(15):run.step()
    with pytest.raises(RuntimeError,match='changed frozen'):run.step()
    assert run.failed and run.db.execute('SELECT count(*) FROM replay_samples').fetchone()[0]==15
    with pytest.raises(RuntimeError,match='unavailable'):run.stop()
    run.close_without_checkpoint()


def test_explicit_contract_cannot_reinterpret_reference_or_research(tmp_path):
    reference=tmp_path/'reference.db';research=tmp_path/'research.db'
    DurableRun(reference,session=session(tmp_path)).stop()
    with pytest.raises(ValueError,match='execution contract'):DurableRun(reference,research_check_interval=16)
    run=DurableRun(research,session=session(tmp_path),research_check_interval=16)
    config=json.loads(run.db.execute('SELECT config FROM replay_meta').fetchone()[0])
    assert config['execution_contract']==RESEARCH_EXECUTION
    run.stop()
    with pytest.raises(ValueError,match='execution contract'):DurableRun(research)
    with pytest.raises(ValueError,match='16-camera'):DurableRun(tmp_path/'invalid.db',research_check_interval=8)


def test_mutation_during_unchecked_recovery_tail_is_guarded_on_exit(tmp_path,monkeypatch):
    path=tmp_path/'run.db';run=DurableRun(path,session=session(tmp_path),research_check_interval=16)
    for _ in range(3):run.step()
    run.close_without_checkpoint()
    original=FirstHitActorEpisode.step
    def mutate_after_record(s,**kwargs):
        record=original(s,**kwargs)
        if s.transitions==3:s.theta[0,0]+=.01
        return record
    monkeypatch.setattr(FirstHitActorEpisode,'step',mutate_after_record)
    with pytest.raises(RuntimeError,match='changed frozen'):
        DurableRun(path,research_check_interval=16)


def test_unsupported_exact_class_rejected_before_initial_snapshot(tmp_path):
    class Unsupported:
        transitions=0
    with pytest.raises(ValueError,match='exact ordinary'):
        DurableRun(tmp_path/'bad.db',session=Unsupported(),research_check_interval=16)
