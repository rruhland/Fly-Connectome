"""Finite sensor interventions: actual camera fixtures, no candidate allocation."""
import copy,sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig
from run_pong_development import validate_camera_job


def session(kind=None,*,episodes=1,ticks=90,warmup=20):
    torch.set_num_threads(1)
    weights=np.random.default_rng(8971).normal(0,.001,(3,4389))
    config=PongEvaluationConfig(99717,episodes=episodes,max_episode_ticks=ticks,opponent_speed=.65)
    mode=None if kind is None else dict(kind=kind,warmup_observations=warmup)
    return FreshPongEvaluator(weights,config,camera_intervention=mode)


def visible_state(value):
    return fingerprint((value.env,value.vision,value.encoder,value.previous,value.latest_encoded,value.policy,value.scaler))


def test_default_live_exact_and_twenty_observation_prefix(monkeypatch):
    live,hold,missing=session(),session('warm_hold'),session('warm_missing')
    explicit=FreshPongEvaluator(live.policy.weights,live.config,camera_intervention=None)
    assert fingerprint(live)==fingerprint(explicit) and not hasattr(live,'camera_intervention')
    captured={};original=type(live.vision).step
    def observe(model,events,frame,**kwargs):
        captured[id(model)]=(events.clone(),None if frame is None else frame.clone())
        return original(model,events,frame,**kwargs)
    monkeypatch.setattr(type(live.vision),'step',observe)
    for transition in range(19):
        a,b,c=live.step(),hold.step(),missing.step()
        assert a==explicit.step()
        assert a=={k:v for k,v in b.items() if k!='camera_observation'}=={k:v for k,v in c.items() if k!='camera_observation'}
        assert visible_state(live)==visible_state(hold)==visible_state(missing)
    assert hold.camera_observations==missing.camera_observations==20
    assert hold.env.game.rally_steps.item()==45
    assert hold.last_observation_mode==missing.last_observation_mode=='live'
    assert hold.held_camera_frame._base is None
    assert hold.held_camera_frame.data_ptr()!=hold.previous.data_ptr()
    held=hold.held_camera_frame.clone()
    a,b,c=live.step(),hold.step(),missing.step()
    # The twentieth action still consumes genuine observation twenty; its NEXT
    # observation is the first masked one. Camera21 can change action21.
    assert a=={k:v for k,v in b.items() if k!='camera_observation'}=={k:v for k,v in c.items() if k!='camera_observation'}
    assert b['camera_observation']==dict(input_mode='live',input_count=20,next_mode='warm_hold',next_count=21)
    assert c['camera_observation']==dict(input_mode='live',input_count=20,next_mode='warm_missing',next_count=21)
    assert fingerprint(live.env)==fingerprint(hold.env)==fingerprint(missing.env)
    assert captured[id(hold.vision)][1] is not None and not captured[id(hold.vision)][0].any()
    assert captured[id(missing.vision)][1] is None and not captured[id(missing.vision)][0].any()
    assert torch.equal(hold.previous,held) and torch.equal(hold.held_camera_frame,held)
    assert hold.latest_encoded['context_available'] and not missing.latest_encoded['context_available']
    assert missing.latest_encoded['context_known'] and missing.latest_encoded['context_age_seconds']==.02
    assert visible_state(live)!=visible_state(hold)!=visible_state(missing)


@pytest.mark.parametrize('kind',['warm_hold','warm_missing'])
@pytest.mark.parametrize('boundary',[19,20])
def test_recovery_before_and_after_mask_onset(tmp_path,kind,boundary):
    run=DurableRun(tmp_path/'camera.db',session=session(kind),snapshot_transitions=1000,snapshot_seconds=100.)
    for _ in range(boundary):run.step()
    at_boundary=fingerprint(run.session)
    expected=run.session.step();expected_state=fingerprint(run.session)
    run.close_without_checkpoint()
    restored=DurableRun(tmp_path/'camera.db')
    assert restored.replayed==boundary and fingerprint(restored.session)==at_boundary
    assert restored.session.camera_observations==boundary+1
    assert restored.session.held_camera_frame.data_ptr()!=restored.session.previous.data_ptr()
    assert restored.step()==expected and fingerprint(restored.session)==expected_state
    restored.stop()
    reopened=DurableRun(tmp_path/'camera.db');assert reopened.replayed==0
    assert fingerprint(reopened.session)==expected_state;reopened.close_without_checkpoint()


@pytest.mark.parametrize('kind',['warm_hold','warm_missing'])
def test_each_episode_resets_warmup_and_cache(kind):
    value=session(kind,episodes=2,ticks=7,warmup=2);rows=[]
    while not value.complete:rows.append(value.step())
    first=[r for r in rows if r['camera_observation']['input_count']==1]
    assert len(first)==2 and [r['episode'] for r in first]==[0,1]
    assert all(r['camera_observation']['input_mode']=='live' for r in first)
    assert all(r['camera_observation']['next_mode']=='live' for r in first)
    assert value.initial_cameras==2 and value.summary()['camera_intervention']['kind']==kind


def test_owned_metadata_frozen_mutation_and_invalid_combinations():
    plain=session();mode=dict(kind='warm_hold',warmup_observations=20)
    value=FreshPongEvaluator(plain.policy.weights,plain.config,camera_intervention=mode)
    mode['warmup_observations']=99
    assert value.camera_intervention['warmup_observations']==20
    value.camera_intervention['kind']='warm_missing'
    with pytest.raises(RuntimeError,match='changed frozen'):value.step()
    for bad in (dict(kind='bad',warmup_observations=20),dict(kind='warm_hold',warmup_observations=True),
                dict(kind='warm_hold',warmup_observations=0),dict(kind='warm_hold',warmup_observations=20,extra=1)):
        with pytest.raises(ValueError,match='camera intervention'):
            FreshPongEvaluator(plain.policy.weights,plain.config,camera_intervention=bad)
    for change in (dict(blank_camera=True),dict(skip_baseline_perception=True),dict(policy_kind='initial')):
        config=PongEvaluationConfig(**{**plain.config.__dict__,**change})
        with pytest.raises(ValueError):FreshPongEvaluator(plain.policy.weights,config,camera_intervention=mode)
    with pytest.raises(ValueError):
        FreshPongEvaluator(plain.policy.weights,plain.config,first_hit_cutoff=True,camera_intervention=mode)


def test_worker_intervention_purpose_is_checked_before_construction():
    good=dict(kind='evaluate',purpose='fresh_evaluation',config=dict(episodes=1,policy_kind='trained'),
        camera_intervention=dict(kind='warm_missing',warmup_observations=20))
    assert validate_camera_job(good)==good['camera_intervention']
    assert validate_camera_job(dict(kind='evaluate')) is None
    for change in (dict(purpose='training_fitness'),dict(kind='learn'),dict(first_hit_cutoff=True),
                   dict(config=dict(episodes=2,policy_kind='trained')),
                   dict(config=dict(episodes=1,policy_kind='trained',blank_camera=True))):
        with pytest.raises(ValueError,match='frozen evaluation'):validate_camera_job(dict(good,**change))
