"""Exact declared coordinates, owned mapping and complete immutable persistence."""
from pathlib import Path
from types import SimpleNamespace
import copy,hashlib,json,random,sqlite3
import numpy as np
import pytest
import torch
from fly_connectome.control.projection import PROJECTION_VERSION,FixedPolicyProjection
from fly_connectome.control.capping import cap_values
from fly_connectome.control.codec import FrozenCodec,fingerprint
from fly_connectome.control.policy import FrozenLinearPolicy
from fly_connectome.control.session import LearningSession
from fly_connectome.control.durable import DurableRun


def descriptor(path,matrix,max_norm2=256.):
    np.save(path,matrix,allow_pickle=False)
    return dict(version=PROJECTION_VERSION,matrix_file=str(path),shape=list(matrix.shape),
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),array_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
        max_norm2=max_norm2,calibration=dict(source='fixed score-free test matrix'))


def encoded(values):
    values=np.asarray(values,dtype=np.float64);indices=np.flatnonzero(values)
    return dict(indices=torch.from_numpy(indices),values=torch.from_numpy(values[indices]))


def test_exact_matrix_cap_after_projection_mapping_blank_and_owned_export(tmp_path):
    matrix=np.array([[1.,2.,0.],[-1.,0.,3.]])
    config=descriptor(tmp_path/'matrix.npy',matrix,max_norm2=4.)
    projection=FixedPolicyProjection(3,config)
    x=np.array([3.,2.,1.]);indices,values=projection.features(encoded(x))
    np.testing.assert_array_equal(values,cap_values(matrix@x,4.))
    assert not np.array_equal(values,matrix@cap_values(x,4.))
    theta=np.array([[1.,-.5],[-1.,.5],[0.,.1]])
    mapped,cap=projection.mapped_evaluation(theta)
    expected=FrozenLinearPolicy(theta).choose(indices,values)[0]
    actual=FrozenLinearPolicy(mapped).choose(np.arange(3),cap_values(x,cap['max_norm2']))[0]
    assert actual==expected
    np.testing.assert_array_equal(mapped,theta@matrix)
    np.testing.assert_array_equal(projection.features(encoded(np.zeros(3)))[1],np.zeros(2))
    cap['calibration']['mapped_projection']['shape'][0]=999
    config['shape'][0]=998
    assert projection.config['shape']==[2,3]
    mapped[0,0]+=10
    np.testing.assert_array_equal(projection.matrix.numpy(),matrix)


def test_declared_artifact_mismatch_and_invalid_sparse_inputs_rejected(tmp_path):
    matrix=np.eye(2,3);config=descriptor(tmp_path/'matrix.npy',matrix)
    wrong=copy.deepcopy(config);wrong['array_sha256']='0'*64
    with pytest.raises(ValueError,match='array shape/dtype/content'):FixedPolicyProjection(3,wrong)
    with pytest.raises(ValueError,match='input dimension'):FixedPolicyProjection(4,config)
    projection=FixedPolicyProjection(3,config)
    for indices,values in [([0,0],[1.,2.]),([3],[1.]),([0],[float('nan')])]:
        with pytest.raises(ValueError):projection.features(dict(indices=torch.tensor(indices),values=torch.tensor(values)))
    config['file_sha256']='0'*64
    with pytest.raises(ValueError,match='file hash'):FixedPolicyProjection(3,config)


def test_immutable_anchor_alias_numpy_mutation_and_path_independent_recovery(tmp_path):
    matrix=np.ones((4,4389),dtype=np.float64);path=tmp_path/'matrix.npy'
    projection=FixedPolicyProjection(4389,descriptor(path,matrix))
    empty=torch.empty((0,2));observer=SimpleNamespace(weights=torch.zeros(2,2),bias=torch.zeros(2))
    memory=SimpleNamespace(keys=empty,values=empty,metric=torch.eye(2))
    vision=SimpleNamespace(observer=observer,state=SimpleNamespace(memory=memory),dynamics={})
    session=SimpleNamespace(vision=vision,projection=projection,matrix_alias=projection.matrix)
    codec=FrozenCodec.from_session(session);payload=codec.dumps(session)
    assert len(payload)<matrix.nbytes/2
    before=fingerprint(session);projection.matrix.numpy()[0,0]+=1
    assert fingerprint(session)!=before
    with pytest.raises(ValueError,match='frozen vision tensors changed'):codec.dumps(session)
    with pytest.raises(ValueError,match='embedded fixed projection changed'):projection.validate_embedded()
    path.unlink()
    restored=codec.loads(payload);restored.projection.validate_embedded()
    assert restored.matrix_alias is restored.projection.matrix
    np.testing.assert_array_equal(restored.projection.matrix.numpy(),matrix)


def test_projected_session_crash_pending_trace_rng_and_full_graph_recovery(tmp_path):
    torch.set_num_threads(1)
    matrix=np.zeros((4,4389),dtype=np.float64)
    matrix[0,4384]=1.;matrix[1,4388]=1.;matrix[2,4253]=1.;matrix[3,0]=1.
    config=descriptor(tmp_path/'matrix.npy',matrix)
    weights=np.random.default_rng(92713).normal(0,.01,(3,4))
    def create(path):
        random.seed(81973);np.random.seed(81973);torch.manual_seed(81973)
        return DurableRun(path,session=LearningSession(187293,max_episode_ticks=25,alpha=.001,lam=.995,
            opponent_speed=.25,weights=weights,policy_projection=config),snapshot_transitions=2)
    a=create(tmp_path/'continuous.db')
    for _ in range(8):a.step()
    expected=fingerprint(a.session);assert a.session.learner.weights.shape==(3,4)
    assert np.count_nonzero(a.session.learner.traces)>0
    assert not np.array_equal(a.session.learner.weights,weights)
    a.stop()
    b=create(tmp_path/'split.db')
    for _ in range(3):b.step()
    b.close_without_checkpoint();b=DurableRun(tmp_path/'split.db');assert b.replayed==1
    for _ in range(5):b.step()
    assert fingerprint(b.session)==expected;b.stop()
    with sqlite3.connect(tmp_path/'continuous.db') as c,sqlite3.connect(tmp_path/'split.db') as d:
        assert c.execute('SELECT * FROM replay_samples ORDER BY seq').fetchall()==d.execute('SELECT * FROM replay_samples ORDER BY seq').fetchall()
