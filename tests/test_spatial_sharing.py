import copy
import numpy as np
import pytest
import torch
from fly_connectome.control.sharing import (SHARING_VERSION,share,unshare,
    FixedSpatialSharing,energy_compensation)
from fly_connectome.control.scaling import make_policy_transform,policy_block_squared_norms
from fly_connectome.control.session import LearningSession
from fly_connectome.control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.codec import fingerprint
from fly_connectome.vision import VisualStateEncoder


def configuration():
    return dict(version=SHARING_VERSION,block_compensation=dict(current=1.,future=1.),
                calibration=dict(seed=849375169,action_seed=686753009,diagnostic=True))


def test_map_invertibility_and_declared_spatial_kernel():
    rng=np.random.default_rng(53271)
    planes=rng.normal(size=(7,16,16))
    np.testing.assert_allclose(unshare(share(planes)),planes,atol=2e-15,rtol=2e-15)
    points=np.zeros((4,16,16));points[0,0,0]=points[1,1,1]=points[2,2,2]=points[3,4,4]=1.
    mapped=share(points).reshape(4,-1)
    np.testing.assert_allclose(np.diag(mapped@mapped.T),1.,atol=1e-15)
    np.testing.assert_allclose((mapped@mapped.T)[0],[1.,.5151515151515151,.1515151515151515,0.],atol=1e-15)
    # Point energy is insufficient for mixtures: calibration must use actual vectors.
    assert np.square(share(np.ones((1,16,16)))).sum()/256==pytest.approx(9/2.0625)


def test_actual_block_energy_compensation_and_raw_transport_unchanged():
    encoder=VisualStateEncoder(64,64,.02);config=configuration()
    indices=torch.tensor([0,1,256,encoder.layout['future'][0],encoder.layout['metadata'][0]])
    values=torch.tensor([1.,2.,.5,3.,4.]);encoded=dict(indices=indices,values=values,layout=encoder.layout)
    before=copy.deepcopy(encoded)
    probe=make_policy_transform(encoder.layout,encoder.dimension,config)
    raw=policy_block_squared_norms(encoded);shared=policy_block_squared_norms(encoded,probe)
    config['block_compensation']=energy_compensation(raw,shared)
    fitted=make_policy_transform(encoder.layout,encoder.dimension,config)
    actual=policy_block_squared_norms(encoded,fitted)
    for name in encoder.layout:assert actual[name]==pytest.approx(raw[name],rel=1e-14)
    assert torch.equal(before['values'],encoded['values']) and torch.equal(before['indices'],encoded['indices'])
    idx,vals=fitted.features(encoded)
    dense=np.zeros(encoder.dimension);dense[idx]=vals
    assert dense[encoder.layout['metadata'][0]]==4.
    assert energy_compensation(dict(current=0,future=0),dict(current=0,future=0))==dict(current=1.,future=1.)


def test_shared_learning_and_frozen_eval_pin_exact_transform(tmp_path):
    torch.set_num_threads(1)
    session=LearningSession(99871,policy_scaling=configuration())
    run=DurableRun(tmp_path/'sharing.db',session=session,snapshot_transitions=3)
    for _ in range(5):run.step()
    state=fingerprint(run.session);weights=run.session.learner.weights.copy();run.close_without_checkpoint()
    restored=DurableRun(tmp_path/'sharing.db')
    assert restored.replayed==2 and fingerprint(restored.session)==state
    assert isinstance(restored.session.scaler,FixedSpatialSharing)
    restored.stop()
    evaluator=FreshPongEvaluator(weights,PongEvaluationConfig(99872,episodes=1,policy_scaling=configuration()))
    evaluator.scaler.config['calibration']['seed']+=1
    with pytest.raises(RuntimeError,match='frozen'):evaluator.step()
