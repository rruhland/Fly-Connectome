"""Focused analytic estimator/boundary/recovery fixtures, no candidate allocation."""
import copy,hashlib,itertools,json
from pathlib import Path
import numpy as np
import pytest
import torch
from fly_connectome.control.pong_actor import probabilities,score_eligibility,batch_update,FirstHitActorEpisode
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.projection import PROJECTION_VERSION
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.durable import DurableRun


def descriptor(tmp_path):
    matrix=np.zeros((2,4389));matrix[0,0]=100.;matrix[1,1]=50.
    path=tmp_path/'matrix.npy';np.save(path,matrix)
    return dict(version=PROJECTION_VERSION,matrix_file=str(path),shape=[2,4389],
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        array_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),max_norm2=256.,calibration={})


def episode(tmp_path):
    torch.set_num_threads(1)
    config=PongEvaluationConfig(710191,episodes=1,max_episode_ticks=30,opponent_speed=.65)
    session=FirstHitActorEpisode(np.array([[.01,-.02],[.02,.01],[-.01,.01]]),config,
        projection=descriptor(tmp_path),temperature=.1,action_seed=598933)
    session.env.game.ball[0]=torch.tensor([.076,.5]);session.env.game.ball_velocity[0]=torch.tensor([-.55,0.])
    return session


def test_enumerated_bandit_loo_is_unbiased_and_analytic_derivative():
    theta=np.array([[.12,-.09],[-.06,.07],[.03,.02]]);x=np.array([.7,-.4]);tau=.4
    _,p=probabilities(theta,x,tau);rewards=np.array([0.,1.,.4]);expected=np.zeros_like(theta)
    for choices in itertools.product(range(3),repeat=4):
        prob=float(np.prod(p[list(choices)]));rs=rewards[list(choices)]
        es=np.array([score_eligibility(a,p,x,tau) for a in choices]);b=(rs.sum()-rs)/3
        expected+=prob*np.einsum('i,ijk->jk',rs-b,es)/4
    direct=sum(p[a]*rewards[a]*score_eligibility(a,p,x,tau) for a in range(3))
    np.testing.assert_allclose(expected,direct,atol=1e-15)
    finite=np.zeros_like(theta);eps=1e-6
    for index in np.ndindex(theta.shape):
        plus=theta.copy();minus=theta.copy();plus[index]+=eps;minus[index]-=eps
        finite[index]=(probabilities(plus,x,tau)[1]@rewards-probabilities(minus,x,tau)[1]@rewards)/(2*eps)
    np.testing.assert_allclose(finite,expected,atol=5e-11)


def test_aggregate_cap_uniform_outcomes_and_partial_rejection():
    theta=np.zeros((3,2));z=np.array([[1.,.5],[.2,-1.]])
    es=np.random.default_rng(9187).normal(size=(4,3,2))
    proposed,d=batch_update(theta,es,[0,1,1,0],z,alpha=1.,step_cap=.01)
    assert d['mixed_outcomes'] and d['nonzero_update'] and d['aggregate_clip_scale']<1
    assert d['applied_update_function_rms']==pytest.approx(.01)
    np.testing.assert_allclose(proposed.sum(axis=0),0,atol=1e-15)
    for rs in ([0]*4,[1]*4):
        got,diag=batch_update(theta,es,rs,z,step_cap=.01)
        assert np.array_equal(got,theta) and not diag['nonzero_update']
    with pytest.raises(ValueError,match='complete'):batch_update(theta,es[:3],[0,1,0],z,step_cap=.01)


def test_same_capped_feature_for_sampling_and_eligibility(tmp_path):
    session=episode(tmp_path);session._begin_episode()
    # Large synthetic generic visual coordinates exercise the projected cap.
    session.latest_encoded={'indices':torch.tensor([0,1]),'values':torch.tensor([10.,-3.])}
    _,x=session.actor_projection.features(session.latest_encoded)
    q,p=probabilities(session.theta,x,session.temperature)
    rng=copy.deepcopy(session.actor_rng);expected_action=int(rng.choice(3,p=p))
    expected_e=score_eligibility(expected_action,p,x,session.temperature)
    action,values=session._choose_policy()
    assert x@x==pytest.approx(256.) and action==expected_action
    assert np.array_equal(q,values) and np.array_equal(expected_e,session.eligibility)
    assert session.last_actor_choice['probabilities']==p.tolist()
    assert session.last_actor_choice['log_probability']==float(np.log(p[action]))
    assert session.decisions==1


def test_terminal_camera_fixed_theta_and_durable_rng_eligibility(tmp_path):
    path=tmp_path/'episode.db';session=episode(tmp_path);theta=session.theta.copy()
    run=DurableRun(path,session=session,snapshot_transitions=100)
    first=run.step();h=fingerprint(run.session);eligibility=run.session.eligibility.copy()
    assert first['actor']['decision']==1 and not first['task_stopped']
    assert np.array_equal(theta,run.session.theta)
    run.close_without_checkpoint();run=DurableRun(path)
    assert run.replayed==1 and fingerprint(run.session)==h
    assert np.array_equal(eligibility,run.session.eligibility)
    expected=copy.deepcopy(run.session);next_record=expected.step()
    last=run.step()
    assert last==next_record and fingerprint(run.session)==fingerprint(expected)
    assert last['task_stopped'] and last['hits']==1 and last['actor']['decision']==2
    assert not last['terminated'] and not last['truncated'] and last['reward']==0
    assert run.session.decisions==run.session.transitions==2 and run.session.complete
    assert torch.equal(run.session.previous,run.session.env.game.render(64,64)[0])
    assert np.array_equal(theta,run.session.theta)
    final=fingerprint(run.session);run.close_without_checkpoint();run=DurableRun(path)
    assert run.replayed==2 and run.session.complete and fingerprint(run.session)==final
    run.stop()


def test_actor_rejects_held_actions_and_parameter_mutation(tmp_path):
    s=episode(tmp_path);config=PongEvaluationConfig(1,episodes=1,action_hold_frames=2)
    with pytest.raises(ValueError,match='K=1'):
        FirstHitActorEpisode(s.theta,config,projection=s.actor_projection.config,temperature=.1,action_seed=1)
    s.theta[0,0]+=.01
    with pytest.raises(RuntimeError,match='changed frozen'):s.step()


def test_actor_checkpoint_requires_exact_external_projection(tmp_path):
    from fly_connectome.control.codec import FrozenCodec
    s=episode(tmp_path);codec=FrozenCodec.from_session(s);payload=codec.dumps(s)
    restored=codec.loads(payload)
    assert fingerprint(restored)==fingerprint(s)
    path=Path(s.actor_projection.config['matrix_file']);original=path.read_bytes()
    path.unlink()
    with pytest.raises(FileNotFoundError):codec.loads(payload)
    path.write_bytes(original+b'tampered')
    with pytest.raises(ValueError,match='hash differs'):codec.loads(payload)
    path.write_bytes(original)
    assert fingerprint(codec.loads(payload))==fingerprint(s)


def test_worker_exports_complete_actor_trace_and_pins_required_artifacts(tmp_path,monkeypatch):
    import sys
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
    import run_pong_development as worker
    import run_dct_sarsa_pong as completion
    from fly_connectome.control.durable import runtime_signature
    from fly_connectome.control.session import source_signature
    projection=descriptor(tmp_path);theta=np.array([[.01,-.02],[.02,.01],[-.01,.01]])
    np.save(tmp_path/'theta.npy',theta)
    matrix=np.load(projection['matrix_file']);np.save(tmp_path/'weights.npy',theta@matrix)
    # Completion pins the common T export name, in addition to its exact descriptor.
    np.save(tmp_path/'search-transform.npy',matrix)
    (tmp_path/'weights.json').write_text(json.dumps(dict(shape=[3,4389],weight_hash=hashlib.sha256((theta@matrix).tobytes()).hexdigest())))
    (tmp_path/'projection.json').write_text(json.dumps(projection))
    config=PongEvaluationConfig(710191,episodes=1,max_episode_ticks=30,opponent_speed=.65)
    from dataclasses import asdict
    job=dict(id='fixture',kind='evaluate',purpose='training_fitness',config=asdict(config),weights_from='weights.npy',
        projection_from='projection.json',first_hit_cutoff=True,compact_reports=True,
        actor=dict(theta_from='theta.npy',temperature=.1,action_seed=598933))
    manifest=dict(source=source_signature(),runtime=runtime_signature(),driver_hash=hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest(),jobs=[job])
    worker.atomic_json(tmp_path/'manifest.json',manifest)
    original=FirstHitActorEpisode.__init__
    def init(self,*args,**kwargs):
        original(self,*args,**kwargs)
        self.env.game.ball[0]=torch.tensor([.076,.5]);self.env.game.ball_velocity[0]=torch.tensor([-.55,0.])
    monkeypatch.setattr(FirstHitActorEpisode,'__init__',init)
    report=worker.worker(tmp_path,'fixture',emit=False)
    assert report['complete'] and report['transitions']==report['actor']['decisions']==2
    completion.pin_completion(tmp_path,job);assert completion.complete_report(tmp_path,job)==json.loads(json.dumps(report))
    pin=json.loads((tmp_path/'fixture-completion.json').read_text())
    assert {'fixture-eligibility.npy','theta.npy','search-transform.npy','projection.json'}<=set(pin['endpoint_artifacts'])
    # Recovery verification makes no new camera observations or eligibility terms.
    verified=worker.worker(tmp_path,'fixture',emit=False,verify_only=True)
    assert verified['new_transitions']==0 and verified['actor']==report['actor']
    completion.pin_completion(tmp_path,job);assert completion.complete_report(tmp_path,job)==json.loads(json.dumps(verified))


def test_contextual_two_start_paired_estimator_exact_expectation():
    theta=np.array([[.12,-.09],[-.06,.07],[.03,.02]]);tau=.4
    contexts=[np.array([.7,-.4]),np.array([-.2,.8])];mass=[.35,.65]
    rewards=[np.array([0.,1.,0.]),np.array([1.,0.,1.])]
    expected=np.zeros_like(theta);truth=np.zeros_like(theta);z=np.eye(2)
    for si,x in enumerate(contexts):
        _,p=probabilities(theta,x,tau)
        truth+=mass[si]*sum(p[a]*rewards[si][a]*score_eligibility(a,p,x,tau) for a in range(3))
    for s0,s1 in itertools.product(range(2),repeat=2):
        xs=[contexts[s0]]*2+[contexts[s1]]*2
        ps=[probabilities(theta,x,tau)[1] for x in xs]
        for actions in itertools.product(range(3),repeat=4):
            prob=mass[s0]*mass[s1]*float(np.prod([p[a] for p,a in zip(ps,actions)]))
            rs=np.array([rewards[s0 if i<2 else s1][a] for i,a in enumerate(actions)])
            es=np.array([score_eligibility(a,p,x,tau) for a,p,x in zip(actions,ps,xs)])
            proposed,diag=batch_update(theta,es,rs,z,alpha=1.,step_cap=100.,paired_starts=True)
            direct=((rs[0]-rs[1])*(es[0]-es[1])+(rs[2]-rs[3])*(es[2]-es[3]))/4.
            np.testing.assert_allclose(proposed-theta,direct,atol=1e-15)
            expected+=prob*(proposed-theta)
    np.testing.assert_allclose(expected,truth,atol=1e-15)


def test_cross_start_difficulty_without_within_start_contrast_is_zero():
    theta=np.zeros((3,2));es=np.random.default_rng(8932).normal(size=(4,3,2));z=np.eye(2)
    got,diag=batch_update(theta,es,[0,0,1,1],z,step_cap=.01,paired_starts=True)
    assert np.array_equal(got,theta) and not diag['nonzero_update']
    assert diag['mixed_outcomes'] and diag['cross_start_only_mixed_batch']
    assert diag['within_start_disagreeing_pairs']==0 and diag['cross_start_disagreeing_pairs']==4
