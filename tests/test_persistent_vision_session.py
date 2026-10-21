"""Optional mutable-vision contract; fixtures are not research allocations."""
import copy
import pickle
import random

import numpy as np
import pytest
import torch

from fly_connectome.control.codec import FrozenCodec, fingerprint, frozen_tensors
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.online_vision import (
    PersistentVisionSession, learned_visual_state, memory_counts,
)
from fly_connectome.control.pong_evaluation import FreshPongEvaluator, PongEvaluationConfig
from fly_connectome.pong import Pong
from fly_connectome.vision import VisualStateEncoder


def weights():
    dimension=VisualStateEncoder(64,64,.02).dimension
    # Fixed, nonzero readout; no task labels or diagnostic oracle coefficients.
    return np.random.default_rng(497103).normal(0,.001,(3,dimension))


def config(**updates):
    options=dict(seed=497107,episodes=2,max_episode_ticks=25,opponent_speed=.65)
    options.update(updates)
    return PongEvaluationConfig(**options)


def rng_hash():
    return fingerprint((random.getstate(),np.random.get_state(),torch.get_rng_state()))


def tensor_storages(value):
    """Check storage ownership rather than merely Python tensor identities."""
    seen=set();storages=set()
    def visit(obj):
        if obj is None or isinstance(obj,(str,bool,int,float,bytes)):return
        if id(obj) in seen:return
        seen.add(id(obj))
        if isinstance(obj,torch.Tensor):
            if obj.numel():storages.add(obj.untyped_storage().data_ptr())
        elif isinstance(obj,dict):
            for k,v in obj.items():visit(k);visit(v)
        elif isinstance(obj,(tuple,list)) or type(obj).__name__=='deque':
            for item in obj:visit(item)
        elif hasattr(obj,'__dict__'):
            visit(vars(obj))
    visit(value)
    return storages


def test_frozen_mode_matches_default_actions_physics_and_visual_state():
    torch.set_num_threads(1)
    before=rng_hash();w=weights();cfg=config()
    old=FreshPongEvaluator(w,cfg)
    new=PersistentVisionSession(w,cfg,learn_enabled=False)
    learned=fingerprint(learned_visual_state(new.vision))
    while not old.complete:
        expected=old.step();actual=new.step()
        metadata=actual.pop('visual_learning')
        assert actual==expected and not metadata['enabled']
        assert fingerprint((old.env,old.vision,old.encoder,old.previous,old.latest_encoded))==fingerprint(
            (new.env,new.vision,new.encoder,new.previous,new.latest_encoded))
    assert fingerprint(learned_visual_state(new.vision))==learned
    assert new.visual_observations==new.transition_cameras+new.initial_cameras
    assert new.visual_learning_observations==0 and rng_hash()==before


def test_online_snapshot_owns_full_banks_calibration_rng_and_scene_reset():
    torch.set_num_threads(1)
    session=PersistentVisionSession(weights(),config(episodes=3),learn_enabled=True)
    before=fingerprint(learned_visual_state(session.vision))
    for _ in range(18):session.step()
    assert fingerprint(learned_visual_state(session.vision))!=before
    # Include every calibration/RNG field even if this short camera fixture did
    # not naturally update a particular horizon/branch yet.
    for horizon,model in session.vision.dynamics.items():
        model.calibration.ranks.append(torch.tensor([.23,.71]))
        session.vision.calibration[horizon].observe(True,.75)
        for branch in ('short','long'):
            bank=getattr(model,branch)
            bank.calibration.ranks.append(torch.tensor([.31,.62]))
            bank.generator.manual_seed(497113+horizon)
            bank.capacity+=1
    learned=fingerprint(learned_visual_state(session.vision))
    snapshot=session.retained_vision();snapshot.validate()
    assert fingerprint(learned_visual_state(snapshot.vision))==learned
    assert not tensor_storages(session.vision)&tensor_storages(snapshot.vision)
    assert snapshot.vision.histories==snapshot.vision.pending==snapshot.vision.forecast_cache=={}
    assert snapshot.vision.state.keys=={} and snapshot.vision.state.tracker.frame==-1
    assert not snapshot.vision.trace.any()
    for horizon,model in session.vision.dynamics.items():
        cloned=snapshot.vision.dynamics[horizon]
        for branch in ('short','long'):
            assert getattr(model,branch).generator is not getattr(cloned,branch).generator
    snap_hash=snapshot.content_hash
    reloaded=pickle.loads(pickle.dumps(snapshot,protocol=5))
    assert reloaded.validate().content_hash==snap_hash
    assert fingerprint(reloaded)==fingerprint(snapshot)
    assert not tensor_storages(reloaded.vision)&tensor_storages(snapshot.vision)
    session.step()
    assert snapshot.validate().content_hash==snap_hash
    assert snapshot.learning_observations==20  # 18 transitions + two episode starts.
    clone=snapshot.clone_model()
    assert not tensor_storages(clone)&tensor_storages(snapshot.vision)
    clone.dynamics[1].short.seen+=1
    assert snapshot.validate().content_hash==snap_hash
    snapshot.vision.dynamics[1].short.calibration.ranks[-1].numpy()[0]+=.1
    with pytest.raises(ValueError,match='source/content'):
        snapshot.clone_model()


def test_retained_frozen_probe_continuation_is_exact_and_cannot_mutate_snapshot():
    torch.set_num_threads(1)
    session=PersistentVisionSession(weights(),config(),learn_enabled=True)
    for _ in range(12):session.step()
    snapshot=session.retained_vision();before=snapshot.content_hash
    first=PersistentVisionSession(weights(),config(),learn_enabled=False,vision_snapshot=snapshot)
    second=PersistentVisionSession(weights(),config(),learn_enabled=False,vision_snapshot=snapshot)
    assert first.starting_visual_snapshot==before
    inherited=snapshot.learning_observations
    assert first.starting_visual_learning_observations==inherited
    assert not tensor_storages(first.vision)&tensor_storages(second.vision)
    while not first.complete:
        assert first.step()==second.step()
        assert fingerprint(first)==fingerprint(second)
    assert snapshot.validate().content_hash==before
    assert first.visual_learning_observations==0
    assert first.retained_vision().learning_observations==inherited
    online=PersistentVisionSession(weights(),config(),learn_enabled=True,vision_snapshot=snapshot)
    online.step()
    assert online.visual_learning_observations==2  # New initial image plus transition.
    assert online.retained_vision().learning_observations==inherited+2
    assert online.summary()['cumulative_visual_learning_observations']==inherited+2
    third=PersistentVisionSession(weights(),config(),learn_enabled=False,vision_snapshot=snapshot)
    third.vision.dynamics[1].long.seen+=1
    with pytest.raises(RuntimeError,match='pinned'):
        third.step()


@pytest.mark.parametrize('crash_count',[12,17])
def test_interrupted_learning_replays_exactly_across_episode_boundaries(tmp_path,crash_count):
    torch.set_num_threads(1)
    # Eleven transitions per episode permit actual short/long learning and
    # horizon-credit settlement. The first crash replays the final episode0
    # camera and episode1's initial/first transition from checkpoint10.
    cfg=PongEvaluationConfig(497127,episodes=3,max_episode_ticks=25,opponent_speed=.65)
    seeds=(497127,497131,497137)
    baseline=PersistentVisionSession(weights(),cfg,learn_enabled=True,episode_seeds=seeds)
    run=DurableRun(tmp_path/'online.db',session=PersistentVisionSession(
        weights(),cfg,learn_enabled=True,episode_seeds=seeds),snapshot_transitions=5)
    initial_counts=memory_counts(run.session.vision)
    for _ in range(crash_count):assert run.step()==baseline.step()
    assert any(len(model.short.keys)>initial_counts['dynamics'][str(h)]['short']
               for h,model in run.session.vision.dynamics.items())
    assert any(model.calibration.ranks for model in run.session.vision.dynamics.values())
    assert any(c.outcomes for c in run.session.vision.calibration.values())
    if crash_count==12:
        checkpoint=run.codec.loads(run.db.execute('SELECT payload FROM replay_active').fetchone()[0])
        assert checkpoint.transitions==10 and checkpoint.vision.pending
        assert checkpoint.episodes==0 and run.session.episodes==1
    else:
        assert run.session.vision.pending
    expected=fingerprint(run.session);counts=memory_counts(run.session.vision)
    run.close_without_checkpoint()
    restored=DurableRun(tmp_path/'online.db')
    assert restored.replayed==crash_count%5 and fingerprint(restored.session)==expected
    assert memory_counts(restored.session.vision)==counts
    while not baseline.complete:
        assert restored.step()==baseline.step()
        assert fingerprint(restored.session)==fingerprint(baseline)
        if restored.session.env.decisions==1:
            seed=seeds[restored.session.episodes-(1 if restored.session.env.needs_reset else 0)]
            assert torch.equal(restored.session.env.game.rng,Pong([seed]).rng)
    assert restored.session.visual_learning_observations==restored.session.transition_cameras+3
    restored.stop()


def test_codec_serializes_mutable_memory_and_pins_observer_without_default_opt_out():
    torch.set_num_threads(1)
    session=PersistentVisionSession(weights(),config(),learn_enabled=True)
    anchors=frozen_tensors(session)
    assert set(anchors)=={'observer.weights','observer.bias'}
    codec=FrozenCodec.from_session(session)
    session.vision.dynamics[1].short.seen+=2
    session.vision.dynamics[1].calibration.ranks.append(torch.tensor([.2,.8]))
    memory=session.vision.state.memory
    memory._variance_keys=memory.keys
    memory._variance=memory.keys.var(0,unbiased=False).clamp(min=1e-4)
    for valid in (True,False):
        memory._variance_version=memory.keys._version if valid else -1
        payload=codec.dumps(session);cloned=codec.loads(payload)
        assert fingerprint(session)==fingerprint(cloned)
        assert not tensor_storages(session.vision)&tensor_storages(cloned.vision)
        other=cloned.vision.state.memory
        assert other._variance_keys is other.keys
        assert (other._variance_version==other._variance_keys._version)==valid
    session.vision.observer.weights.numpy()[0,0,0,0]+=.1
    with pytest.raises(ValueError,match='frozen vision tensors'):
        codec.dumps(session)
    default=FreshPongEvaluator(weights(),config())
    default.visual_memory_contract=session.visual_memory_contract
    with pytest.raises(ValueError,match='unknown mutable'):
        frozen_tensors(default)


def test_snapshot_source_and_restored_learning_mode_tampering_fail_closed():
    torch.set_num_threads(1)
    session=PersistentVisionSession(weights(),config(),learn_enabled=True)
    snapshot=session.retained_vision()
    snapshot.source['control/online_vision.py']='wrong'
    assert session.source['control/online_vision.py']!='wrong'
    with pytest.raises(ValueError,match='source/content'):
        PersistentVisionSession(weights(),config(),learn_enabled=True,vision_snapshot=snapshot)
    wrong=copy.deepcopy(session);wrong.learn_enabled=False
    with pytest.raises(ValueError,match='parameters changed'):
        PersistentVisionSession.validate_restored(wrong)


@pytest.mark.parametrize('updates',[{'blank_camera':True},{'action_hold_frames':2},
                                      {'policy_kind':'noop'},{'skip_baseline_perception':True}])
def test_mutable_session_rejects_ambiguous_sensor_or_control_modes(updates):
    with pytest.raises(ValueError):
        PersistentVisionSession(weights(),config(**updates),learn_enabled=True)
