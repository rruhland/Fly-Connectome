"""Fixed pre-action control variate and analytic all-past regression checks."""
import itertools
import copy
import hashlib
import numpy as np
import pytest
import torch
from fly_connectome.control.pong_actor import (probabilities,score_eligibility,baseline_prediction,
    baseline_statistics,fit_baseline,BASELINE_RIDGE,batch_update)
from fly_connectome.control.pong_actor import FirstHitActorEpisode
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.projection import PROJECTION_VERSION
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.codec import fingerprint


def test_weighted_centered_ridge_intercept_and_zero_weight():
    a,h,w=baseline_statistics(np.zeros((2,2)),np.array([1.,1.]),1)
    b,k,v=baseline_statistics(np.zeros((1,2)),np.array([1.]),0)
    beta=fit_baseline(a+b,h+k,w+v)
    np.testing.assert_allclose(beta,[1/6/(1+BASELINE_RIDGE),0,0],atol=1e-15)
    assert baseline_prediction(beta,np.zeros(2))[0]==.5+beta[0]
    assert baseline_prediction([2.,0,0],np.zeros(2))==(1.,2.5)
    assert np.array_equal(fit_baseline(np.zeros((3,3)),np.zeros(3),0),np.zeros(3))
    with pytest.raises(ValueError,match='zero-weight'):fit_baseline(np.eye(3),np.zeros(3),0)


def test_production_score_baseline_unbiased_with_correlated_latent_bias_and_early_stop():
    theta=np.array([[.1,-.2],[-.1,.03],[.05,.08]]);beta=np.array([.1,.5,-.2]);tau=.35
    def expectation(head):
        raw=np.zeros_like(head);controlled=np.zeros_like(head);objective=0.
        for s0,a0 in itertools.product((-1,1),range(3)):
            x=np.array([.7,-.3]);q,p=probabilities(head,x,tau,np.array([.2*s0,0,-.2*s0]))
            z=score_eligibility(a0,p,x,tau);v=baseline_prediction(beta,x)[0];pr=.5*p[a0]
            if a0==0:
                reward=.3;objective+=pr*reward;raw+=pr*reward*z;controlled+=pr*(reward*z-v*z);continue
            for s1,a1 in itertools.product((-1,1),range(3)):
                x1=np.array([.2+.1*a0,.8]);q1,p1=probabilities(head,x1,tau,np.array([.2*s1,0,-.2*s1]))
                z1=score_eligibility(a1,p1,x1,tau);v1=baseline_prediction(beta,x1)[0]
                path=pr*(.8 if s1==s0 else .2)*p1[a1];reward=float(a1==2)
                objective+=path*reward;raw+=path*reward*(z+z1)
                controlled+=path*(reward*(z+z1)-v*z-v1*z1)
        return objective,raw,controlled
    objective,raw,controlled=expectation(theta)
    np.testing.assert_allclose(controlled,raw,atol=3e-16)
    numerical=np.zeros_like(theta);eps=1e-6
    for index in np.ndindex(theta.shape):
        plus=theta.copy();minus=theta.copy();plus[index]+=eps;minus[index]-=eps
        numerical[index]=(expectation(plus)[0]-expectation(minus)[0])/(2*eps)
    np.testing.assert_allclose(controlled,numerical,atol=1e-10)


def test_state_update_and_uniform_pair_noise_are_explicit():
    theta=np.zeros((3,2));es=np.arange(24,dtype=float).reshape(4,3,2)/10
    cs=es*.5;z=np.eye(2);rewards=[1,1,1,1]
    paired,dp=batch_update(theta,es,rewards,z,step_cap=1.,paired_starts=True)
    state,ds=batch_update(theta,es,rewards,z,step_cap=1.,paired_starts=True,baseline_corrections=cs)
    assert np.array_equal(paired,theta) and not dp['nonzero_update']
    expected=(es-cs).sum(axis=0)/4*1e-5;expected-=expected.mean(axis=0,keepdims=True)
    np.testing.assert_array_equal(state,expected)
    assert ds['baseline']=='lagged-visual-state-mc-v1' and ds['baselines'] is None and ds['nonzero_update']


def test_baseline_correction_features_and_rng_exact_after_unchecked_tail(tmp_path):
    torch.set_num_threads(1);matrix=np.zeros((2,4389));matrix[0,0]=100.;matrix[1,1]=50.
    path=tmp_path/'matrix.npy';np.save(path,matrix)
    projection=dict(version=PROJECTION_VERSION,matrix_file=str(path),shape=[2,4389],max_norm2=256.,calibration={},
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),array_sha256=hashlib.sha256(matrix.tobytes()).hexdigest())
    original=FirstHitActorEpisode(np.array([[.01,-.02],[.02,.01],[-.01,.01]]),
        PongEvaluationConfig(710191,episodes=1,max_episode_ticks=80,opponent_speed=.65),projection=projection,temperature=.1,
        action_seed=598933,noise=dict(kind='centered-gaussian-ar1-v1',sigma=.06,rho=.92,seed=827193),baseline=np.array([.1,.2,-.05]))
    reference=copy.deepcopy(original);run=DurableRun(tmp_path/'run.db',session=original,research_check_interval=16)
    for _ in range(3):assert reference.step()==run.step()
    expected=fingerprint(reference);run.close_without_checkpoint();run=DurableRun(tmp_path/'run.db',research_check_interval=16)
    assert run.replayed==3 and fingerprint(run.session)==expected
    while not reference.complete:assert reference.step()==run.step()
    assert fingerprint(run.session)==fingerprint(reference)
    np.testing.assert_array_equal(run.session.baseline_correction,reference.baseline_correction)
    np.testing.assert_array_equal(run.session.baseline_features,reference.baseline_features)
    run.session.actor_baseline[0]+=.1
    with pytest.raises(RuntimeError,match='changed frozen'):run.stop()
    run.close_without_checkpoint()
