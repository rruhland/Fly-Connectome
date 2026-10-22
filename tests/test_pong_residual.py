"""Focused residual estimator, exact warm-start and durable recovery checks."""
import copy,hashlib,itertools,json
import numpy as np
import pytest
import torch
from fly_connectome.control.pong_actor import FirstHitActorEpisode,probabilities,score_eligibility
from fly_connectome.control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig
from fly_connectome.control.pong_residual import FixedResidualMap,ResidualFirstHitActorEpisode,ProjectedResidualGreedy,residual_probabilities,residual_batch_update,combined_scores
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.durable import DurableRun
from test_pong_actor import descriptor


def fixed_map(tmp_path,d=2):
    rng=np.random.default_rng(193713);A=rng.standard_normal((64,d));c=rng.uniform(-1,1,64)
    path=tmp_path/'residual.npz';np.savez(path,A=A,c=c)
    ah=lambda a:hashlib.sha256(a.tobytes()).hexdigest()
    return dict(kind='fixed-tanh-residual-v1',file=str(path),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),A_sha256=ah(A),c_sha256=ah(c),seed=193713,shape=[64,d])


def actor(tmp_path,B=None):
    torch.set_num_threads(1)
    return ResidualFirstHitActorEpisode(np.array([[.01,-.02],[.02,.01],[-.01,.01]]),np.zeros((3,64)) if B is None else B,
        PongEvaluationConfig(710191,episodes=1,max_episode_ticks=80,opponent_speed=.65),projection=descriptor(tmp_path),
        residual_map=fixed_map(tmp_path),temperature=.1,action_seed=598933,
        noise=dict(kind='centered-gaussian-ar1-v1',sigma=.06,rho=float(np.exp(-.02/.25)),seed=827193))


def test_exact_zero_residual_scores_probabilities_actions_and_trajectory(tmp_path):
    s=actor(tmp_path);old=FirstHitActorEpisode(s.theta,s.config,projection=s.actor_projection.config,temperature=s.temperature,action_seed=s.actor_action_seed,noise=s.actor_bias_config)
    for x in (np.zeros(2),np.array([12.,-8.]),np.array([-.2,.3])):
        q,p=probabilities(s.theta,x,s.temperature,s.bias)
        got,prob,h=residual_probabilities(s.theta,s.B,x,s.residual_map,s.temperature,s.bias)
        assert np.array_equal(q,got) and np.array_equal(p,prob) and h@h<=64
    while not old.complete:
        assert old.step()==s.step()
        assert np.array_equal(old.eligibility,s.eligibility)
        assert old.actor_rng.bit_generator.state==s.actor_rng.bit_generator.state
        assert old.bias_rng.bit_generator.state==s.bias_rng.bit_generator.state
    assert np.any(s.residual_eligibility)


def test_enumerated_conditional_two_branch_score_and_finite_difference(tmp_path):
    mapping=FixedResidualMap(fixed_map(tmp_path),2);theta=np.array([[.1,-.2],[-.2,.1],[.03,.01]])
    B=np.random.default_rng(173).normal(0,.03,(3,64));x=np.array([.7,-.4]);tau=.4
    rewards=np.array([0.,1.,.4]);expectedT=np.zeros_like(theta);expectedB=np.zeros_like(B)
    # Exogenous finite latent-bias mixture, condition on its actual probabilities.
    for bias in (np.array([.15,-.05,-.1]),np.array([-.15,.05,.1])):
        q,p,h=residual_probabilities(theta,B,x,mapping,tau,bias)
        for a in range(3):
            expectedT+=.5*p[a]*rewards[a]*score_eligibility(a,p,x,tau)
            expectedB+=.5*p[a]*rewards[a]*score_eligibility(a,p,h,tau)
    def objective(t,b):return sum(.5*(residual_probabilities(t,b,x,mapping,tau,sign*np.array([.15,-.05,-.1]))[1]@rewards) for sign in (-1,1))
    eps=1e-6
    for parameter,expected in ((theta,expectedT),(B,expectedB)):
        indices=list(np.ndindex(parameter.shape)) if parameter is theta else [(0,0),(1,17),(2,63)]
        for index in indices:
            plus=parameter.copy();minus=parameter.copy();plus[index]+=eps;minus[index]-=eps
            fd=(objective(plus,B)-objective(minus,B))/(2*eps) if parameter is theta else (objective(theta,plus)-objective(theta,minus))/(2*eps)
            assert fd==pytest.approx(expected[index],abs=1e-10)


def test_capped_features_same_sampling_greedy_and_both_eligibilities(tmp_path):
    B=np.random.default_rng(1121).normal(0,.1,(3,64));s=actor(tmp_path,B);s._begin_episode()
    s.latest_encoded={'indices':torch.tensor([0,1]),'values':torch.tensor([10.,-3.])}
    _,x=s.actor_projection.features(s.latest_encoded);assert x@x==pytest.approx(256.)
    q,p,h=residual_probabilities(s.theta,s.B,x,s.residual_map,s.temperature,s.bias)
    expected=int(copy.deepcopy(s.actor_rng).choice(3,p=p));a,got=s._choose_policy()
    assert a==expected and np.array_equal(got,q)
    assert np.array_equal(s.eligibility,score_eligibility(a,p,x,s.temperature))
    assert np.array_equal(s.residual_eligibility,score_eligibility(a,p,h,s.temperature))
    greedy=ProjectedResidualGreedy(s.theta,s.B,s.config,projection=s.actor_projection.config,residual_map=s.residual_map.config)
    greedy.latest_encoded=s.latest_encoded
    ga,gq=greedy._choose_policy();assert ga==np.argmax(q) and np.array_equal(gq,q)


def test_legacy_greedy_actions_equivalent_but_score_scale_separate(tmp_path):
    s=actor(tmp_path);g=ProjectedResidualGreedy(s.theta,s.B,s.config,projection=s.actor_projection.config,residual_map=s.residual_map.config)
    old=FreshPongEvaluator(s.policy.weights,s.config)
    while not old.complete:
        a=old.step();b=g.step()
        for key in ('action','reward','hits','terminated','truncated','elapsed_seconds','physics_ticks'):assert a[key]==b[key]
    assert g.transitions==old.transitions


@pytest.mark.parametrize('kind',['actor','greedy'])
def test_residual_exact_research_tail_recovery_and_parameter_anchors(tmp_path,kind):
    s=actor(tmp_path,np.random.default_rng(119).normal(0,.01,(3,64)))
    if kind=='greedy':s=ProjectedResidualGreedy(s.theta,s.B,s.config,projection=s.actor_projection.config,residual_map=s.residual_map.config)
    run=DurableRun(tmp_path/'residual.db',session=s,research_check_interval=16)
    for _ in range(3):run.step()
    h=fingerprint(run.session);run.close_without_checkpoint()
    run=DurableRun(tmp_path/'residual.db',research_check_interval=16)
    assert run.replayed==3 and fingerprint(run.session)==h
    expected=copy.deepcopy(run.session)
    assert run.step()==expected.step() and fingerprint(run.session)==fingerprint(expected)
    run.session.B[0,0]+=.1
    with pytest.raises(RuntimeError,match='changed frozen'):run.session.verify_frozen()
    run.close_without_checkpoint()


def test_combined_aggregate_update_cap_uniform_and_partial(tmp_path):
    mapping=FixedResidualMap(fixed_map(tmp_path),2);rng=np.random.default_rng(118)
    theta=rng.normal(size=(3,2));B=np.zeros((3,64));z=rng.normal(size=(12,2));h=np.array([mapping.features(x) for x in z])
    es=rng.normal(size=(4,3,2));fs=rng.normal(size=(4,3,64))
    t,b,d=residual_batch_update(theta,B,es,fs,[0,1,1,0],z,h,alpha=1.,step_cap=.01)
    assert d['aggregate_clip_scale']<1 and d['applied_update_function_rms']==pytest.approx(.01)
    assert np.sqrt(np.mean(combined_scores(t-theta,b-B,z,h)**2))==pytest.approx(.01)
    for rewards in ([0]*4,[1]*4):
        t,b,d=residual_batch_update(theta,B,es,fs,rewards,z,h,alpha=1.,step_cap=.01)
        assert np.array_equal(t,theta) and np.array_equal(b,B)
    with pytest.raises(ValueError,match='complete'):residual_batch_update(theta,B,es[:3],fs[:3],[0,1,0],z,h,alpha=1.,step_cap=.01)


def test_map_content_and_parameter_mutation_rejected(tmp_path):
    s=actor(tmp_path);s.residual_map.A[0,0]+=.1
    with pytest.raises(ValueError,match='map changed'):s.residual_map.validate_embedded()
    with pytest.raises(RuntimeError,match='changed frozen'):s.verify_frozen()
    config=s.residual_map.config
    with open(config['file'],'ab') as stream:stream.write(b'tamper')
    with pytest.raises(ValueError,match='file hash'):FixedResidualMap(config,2)
