import time
import numpy as np
from pong_relational_policy_adapter import from_payload,to_payload,model_hash,new_controller,choose,observe
from pong_relational_feedback import RelationalPolicy,structural_choices
from stateful_reward_fitness import CameraBudget,fitness_key,evaluate
from run_pong_relational_search import proposals,complete_score,rank_complete,BASELINE_ID,unique_population


def test_first_population_covers_all_supported_bindings_and_screen_ties_are_stable():
    p=proposals(0,None,np.ones(4),np.random.default_rng(42))
    expected={b for b in structural_choices() if b[0] in (2,3)}
    assert len(p)==32 and {tuple(x['choices']) for x in p}==expected
    assert {tuple(x['choices']) for x in p if not np.any(x['coefficients'])}==expected
    assert rank_complete([('first',3),('second',3),('third',4)])==[('third',4),('first',3),('second',3)]


def test_four_seed_fitness_cannot_be_mistaken_for_eight_or_sixteen():
    payload=proposals(0,None,np.ones(4),np.random.default_rng(1))[0];identity='fixed'
    cache={fitness_key(model_hash(payload),s,identity):dict(own_hits=2) for s in range(4)}
    spec=dict(identity=identity)
    assert complete_score(payload,list(range(4)),cache,spec)==8
    assert complete_score(payload,list(range(8)),cache,spec) is None
    assert complete_score(payload,list(range(16)),cache,spec) is None
    for s in range(16):cache[fitness_key(BASELINE_ID,s,identity)]=dict(own_hits=1)
    assert complete_score(None,list(range(16)),cache,spec)==16


def test_adapter_hysteresis_uses_actual_baseline_and_fallback_commands():
    p=RelationalPolicy(2,0,1,0,1,[0,0,0,0,2],np.ones(4));c=new_controller(p)
    scene=np.array([[.5,.1,0,0,1,0],[.5,.9,0,0,1,0]])
    a,_=choose(c,scene,2,np.zeros(3));assert a==2
    observe(c,dict(hits=1,terminated=False))
    a,_=choose(c,scene,1,np.zeros(3));assert a==2
    observe(c,dict(hits=0,terminated=False))
    a,_=choose(c,scene[:1],0,np.zeros(3));assert a==0
    observe(c,dict(hits=0,terminated=False))
    a,_=choose(c,scene,1,np.zeros(3));assert a==0
    assert model_hash(to_payload(p))==model_hash(to_payload(from_payload(to_payload(p))))


def test_expired_generic_job_has_no_camera_and_budget_keeps_inflight_grants(tmp_path,monkeypatch):
    import stateful_reward_fitness as worker
    def forbidden(*args,**kwargs):raise AssertionError('expired job must not construct a native evaluator')
    monkeypatch.setattr(worker,'RewardEvaluator',forbidden)
    payload=to_payload(RelationalPolicy(2,0,1,0,1,np.zeros(5),np.ones(4)))
    r=evaluate('pong_relational_policy_adapter',payload,1,tmp_path,time.monotonic()-1,1000,'fixed')
    assert r['not_started'] and r['transitions']==r['initial_cameras']==0
    b=CameraBudget(2000);assert b.reserve(1000) and b.reserve(1000) and not b.reserve(1000)
    b.finish(1000,dict(transitions=123,initial_cameras=1));assert b.used+b.reserved==1123


def test_round_dedup_preserves_first_priority_and_previous_round_exclusions():
    population=proposals(0,None,np.ones(4),np.random.default_rng(2));a,b,c=population[:3]
    seen={model_hash(c)}
    assert unique_population([a,b,a,c,b],seen)==[a,b]
    assert model_hash(a) in seen and model_hash(b) in seen


def test_exact_copy_installs_only_before_camera_and_preserves_frozen_fields():
    from exact_visual_transport_copy import install_exact_transport_copy,ExactCopyEncoder
    from fly_connectome.vision.interface import VisualStateEncoder
    from fly_connectome.control.codec import fingerprint
    class Session:
        transitions=0;initial_cameras=0
        def __init__(self):
            self.encoder=VisualStateEncoder(64,64,.02);self.initial_learned_hash=self.learned_hash()
        def learned_hash(self):return fingerprint(vars(self.encoder))
    session=Session();before=session.learned_hash();r=install_exact_transport_copy(session)
    assert type(session.encoder) is ExactCopyEncoder and session.learned_hash()==before
    assert r['identical_encoder_code_object'] and r['frozen_hash_before']==r['frozen_hash_after']
    late=Session();late.transitions=1
    import pytest
    with pytest.raises(ValueError):install_exact_transport_copy(late)
