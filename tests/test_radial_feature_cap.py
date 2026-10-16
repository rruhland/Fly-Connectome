import copy
import numpy as np
import pytest
import torch
from fly_connectome.control.capping import CAP_VERSION,cap_values
from fly_connectome.control.scaling import make_policy_transform,policy_features
from fly_connectome.control.sarsa import TrueOnlineSarsa
from fly_connectome.control.policy import FrozenLinearPolicy
from fly_connectome.control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig
from fly_connectome.control.session import LearningSession
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.durable import DurableRun
from fly_connectome.vision import VisualStateEncoder


def configuration():
    return dict(version=CAP_VERSION,max_norm2=256.,base_transform=None,
                calibration=dict(seed=849375169,transitions=1024,fixed=True))


def test_radial_cap_is_identity_below_bound_and_handles_extreme_finite_values():
    small=np.array([3.,4.],dtype=np.float32)
    assert cap_values(small,256.) is small
    for values in (np.array([300.,-400.,0.]),np.array([1e308,-1e308,0.])):
        actual=cap_values(values,256.)
        assert np.isfinite(actual).all() and actual@actual==pytest.approx(256.)
        assert actual[-1]==0 and actual[0]>0>actual[1]
    with pytest.raises(ValueError,match='finite'):cap_values(np.array([np.inf]),256.)


def test_fixed_cap_preserves_raw_values_sparsity_and_greedy_argmax():
    encoder=VisualStateEncoder(64,64,.02)
    encoded=dict(indices=torch.tensor([0,10,encoder.dimension-1]),values=torch.tensor([30.,40.,2.]),layout=encoder.layout)
    before=copy.deepcopy(encoded);cap=make_policy_transform(encoder.layout,encoder.dimension,configuration())
    idx,values=policy_features(encoded,cap)
    np.testing.assert_array_equal(idx,encoded['indices'])
    assert values@values==pytest.approx(256.)
    assert torch.equal(encoded['values'],before['values'])
    rng=np.random.default_rng(99881)
    for _ in range(20):
        policy=FrozenLinearPolicy(rng.normal(size=(3,encoder.dimension)))
        assert policy.choose(encoded['indices'],encoded['values'])[0]==policy.choose(idx,values)[0]


def test_bounded_inputs_make_dutch_trace_operator_contractive():
    rng=np.random.default_rng(99882)
    for _ in range(30):
        x=cap_values(rng.normal(size=12)*10**rng.uniform(-2,8),256.)
        beta=.995;operator=beta*(np.eye(len(x))-.001*np.outer(x,x))
        assert np.linalg.norm(operator,ord=2)<=beta+1e-14


def test_nonfinite_update_rejected_before_weight_mutation():
    learner=TrueOnlineSarsa(1,1,alpha=2.,epsilon=0)
    learner.start(np.array([0]),np.array([1e308]))
    original=learner.weights
    with pytest.raises(FloatingPointError):learner.advance(1.,.02,terminated=True)
    assert learner.weights is original and learner.weights[0,0]==0.
    with pytest.raises(FloatingPointError):FrozenLinearPolicy([[1e308]]).q_values([0],[1e308])


def test_cap_configuration_is_frozen_and_durable_replay_exact(tmp_path):
    torch.set_num_threads(1)
    session=LearningSession(99883,opponent_speed=.25,policy_scaling=configuration())
    run=DurableRun(tmp_path/'capped.db',session=session,snapshot_transitions=3)
    for _ in range(5):run.step()
    state=fingerprint(run.session);run.close_without_checkpoint()
    restored=DurableRun(tmp_path/'capped.db');assert restored.replayed==2 and fingerprint(restored.session)==state
    restored.stop()
    evaluator=FreshPongEvaluator(np.zeros((3,session.encoder.dimension)),PongEvaluationConfig(99884,policy_scaling=configuration()))
    evaluator.scaler.max_norm2=128.
    with pytest.raises(RuntimeError,match='frozen'):evaluator.step()


def test_failed_update_preserves_last_valid_journal_not_advanced_live_session(tmp_path):
    torch.set_num_threads(1)
    session=LearningSession(99885,alpha=1e308)
    # Fault fixture: immediate loss with an intentionally unsafe finite step size.
    session.env.game.ball[0]=torch.tensor([-.01,.05])
    session.env.game.ball_velocity[0]=torch.tensor([-.55,0.])
    before=fingerprint(session)
    run=DurableRun(tmp_path/'failed-update.db',session=session)
    with pytest.raises(FloatingPointError):run.step()
    assert run.failed and run.db.execute('SELECT count(*) FROM replay_samples').fetchone()[0]==0
    with pytest.raises(RuntimeError,match='unavailable'):run.stop()
    run.close_without_checkpoint()
    recovered=DurableRun(tmp_path/'failed-update.db')
    assert recovered.session.transitions==0 and fingerprint(recovered.session)==before
    assert np.isfinite(recovered.session.learner.weights).all()
    recovered.stop()
