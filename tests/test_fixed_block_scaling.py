import copy
import numpy as np
import pytest
import torch
from fly_connectome.control.scaling import (SCALING_VERSION,energy_matched_scales,
    FixedBlockScaling,block_squared_norms)
from fly_connectome.control.toy_session import ToySession,ToySpec
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.fresh_evaluation import FreshFrozenEvaluator,EvaluationConfig
from fly_connectome.vision import VisualStateEncoder


def configuration():
    encoder=VisualStateEncoder(64,64,.02)
    return dict(version=SCALING_VERSION,scales={name:2. for name in encoder.layout},
                calibration=dict(seed=99851,transitions=0,diagnostic=True))


def test_energy_preserved_with_bounded_scales_and_absent_block():
    means=dict(current=7.,future=6.,future_outside=0.,relations=8.,context=.15,metadata=56.)
    scales=energy_matched_scales(means)
    assert scales['future_outside']==1 and all(.125<=s<=8 for s in scales.values())
    assert scales['context']==8
    assert sum(means[k]*scales[k]**2 for k in means)==pytest.approx(sum(means.values()),rel=1e-14)
    assert scales==energy_matched_scales(means)
    assert energy_matched_scales({'missing':0})=={'missing':1.}


def test_boundary_transform_preserves_raw_state_sparse_zeros_and_count_ratios():
    encoder=VisualStateEncoder(64,64,.02)
    scaler=FixedBlockScaling(encoder.layout,encoder.dimension,configuration())
    encoded=dict(indices=torch.tensor([0,1,encoder.dimension-1]),values=torch.tensor([0.,1.,3.]),layout=encoder.layout)
    before=copy.deepcopy(encoded)
    indices,values=scaler.features(encoded)
    np.testing.assert_array_equal(indices,[0,1,encoder.dimension-1])
    np.testing.assert_array_equal(values,[0.,2.,6.])
    assert torch.equal(encoded['values'],before['values'])
    assert block_squared_norms(encoded)['metadata']==9


def test_scaled_toy_starts_with_transformed_policy_features_only():
    torch.set_num_threads(1)
    session=ToySession(ToySpec('delayed',99852,2,diagnostic=True,policy_scaling=configuration()))
    encoded=session.latest_encoded
    np.testing.assert_array_equal(session.learner.pending_features[encoded['indices']],2*encoded['values'].numpy())
    assert session.learner.weights.sum()==0
    with pytest.raises(ValueError,match='unscaled'):
        ToySpec('immediate',2101,200,policy_scaling=configuration()).validate()


def test_frozen_evaluation_pins_exact_fit_config_and_multipliers():
    torch.set_num_threads(1)
    dimension=VisualStateEncoder(64,64,.02).dimension
    evaluator=FreshFrozenEvaluator(np.zeros((2,dimension)),EvaluationConfig('delayed',99853,2),policy_scaling=configuration())
    evaluator.scaler.config['calibration']['seed']+=1
    with pytest.raises(RuntimeError,match='frozen'):
        evaluator.step()


def test_scaled_mid_delay_state_replays_with_exact_fit_configuration(tmp_path):
    torch.set_num_threads(1)
    session=ToySession(ToySpec('delayed',99854,2,diagnostic=True,policy_scaling=configuration()))
    run=DurableRun(tmp_path/'scaled.db',session=session,snapshot_transitions=3)
    for _ in range(5):run.step()
    end=fingerprint(run.session);run.close_without_checkpoint()
    restored=DurableRun(tmp_path/'scaled.db')
    assert restored.replayed==2 and fingerprint(restored.session)==end
    assert restored.session.scaler.config==configuration()
    assert restored.session.scaler.configuration_hash==session.scaler.configuration_hash
    restored.stop()
