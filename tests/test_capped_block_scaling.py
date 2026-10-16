import copy
import numpy as np
import pytest
import torch
from fly_connectome.control.capping import CAP_VERSION,cap_values
from fly_connectome.control.scaling import SCALING_VERSION,make_policy_transform,policy_features
from fly_connectome.control.session import LearningSession
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig
from fly_connectome.vision import VisualStateEncoder


def configuration():
    names=VisualStateEncoder(64,64,.02).layout
    return dict(version=CAP_VERSION,max_norm2=256.,calibration={'fixed':True},
        base_transform=dict(version=SCALING_VERSION,calibration={'fixed':True},
            scales={name:8. if name=='context' else .5 for name in names}))


def test_block_conditioning_precedes_cap_and_preserves_raw_transport():
    encoder=VisualStateEncoder(64,64,.02);context=encoder.layout['context'][0]
    encoded=dict(indices=torch.tensor([0,context]),values=torch.tensor([20.,20.]),layout=encoder.layout)
    before=copy.deepcopy(encoded);config=configuration()
    transform=make_policy_transform(encoder.layout,encoder.dimension,config)
    indices,actual=policy_features(encoded,transform)
    np.testing.assert_array_equal(indices,encoded['indices'])
    np.testing.assert_array_equal(actual,cap_values(np.array([10.,160.]),256.))
    assert actual@actual==pytest.approx(256.)
    assert not np.allclose(actual,cap_values(encoded['values'],256.)*np.array([.5,8.]))
    assert torch.equal(encoded['values'],before['values'])


def test_composed_transform_recovers_exactly_and_frozen_evaluation_checks_base(tmp_path):
    torch.set_num_threads(1);config=configuration()
    session=LearningSession(99886,opponent_speed=.25,policy_scaling=config)
    run=DurableRun(tmp_path/'composition.db',session=session,snapshot_transitions=3)
    for _ in range(5):run.step()
    state=fingerprint(run.session);run.close_without_checkpoint()
    restored=DurableRun(tmp_path/'composition.db')
    assert restored.replayed==2 and fingerprint(restored.session)==state
    assert restored.session.scaler.base.config==config['base_transform']
    restored.stop()
    evaluator=FreshPongEvaluator(np.zeros((3,session.encoder.dimension)),PongEvaluationConfig(99887,policy_scaling=config))
    evaluator.scaler.base.multipliers.flags.writeable=True
    evaluator.scaler.base.multipliers[0]*=2
    with pytest.raises(RuntimeError,match='frozen'):evaluator.step()
