import random
import numpy as np
import pytest
import torch
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.pong_evaluation import FreshPongEvaluator, PongEvaluationConfig
from fly_connectome.control.sarsa import TrueOnlineSarsa
from fly_connectome.control.session import LearningSession
from fly_connectome.pong import Pong
from fly_connectome.vision import VisualStateEncoder


def rng_hash():
    return fingerprint((random.getstate(),np.random.get_state(),torch.get_rng_state()))


def test_constructor_uses_first_serve_and_evaluation_is_fresh_and_isolated(monkeypatch):
    torch.set_num_threads(1)
    session=LearningSession(99710,max_episode_ticks=3)
    session.learner.weights[:] = .001
    before,globals_before=fingerprint(session),rng_hash()
    config=PongEvaluationConfig(99711,episodes=2,max_episode_ticks=3)
    first=FreshPongEvaluator(session.learner.weights,config)
    assert torch.equal(first.env.game.rng,Pong([99711]).rng)
    assert first.encoder.sample is None and first.vision.state.tracker.frame == -1
    assert not first.previous.any() and first.latest_encoded is None
    def forbidden(*args,**kwargs):raise AssertionError('learner update called during evaluation')
    monkeypatch.setattr(TrueOnlineSarsa,'advance',forbidden)
    rows=[]
    while not first.complete:rows.append(first.step())
    assert any(v != 0 for r in rows for v in r['q'])
    first.previous.fill_(True);first.vision.histories[999]=[]
    second=FreshPongEvaluator(session.learner.weights,config)
    actual=[]
    while not second.complete:actual.append(second.step())
    assert rows == actual
    assert fingerprint(session)==before and rng_hash()==globals_before
    assert second.summary()['truncations']==2 and second.summary()['win_fraction']==0
    assert second.initial_cameras == 2


@pytest.mark.parametrize('kind',['initial','random','noop','up'])
def test_skipping_baseline_perception_preserves_actions_physics_and_serves(kind):
    torch.set_num_threads(1)
    options=dict(seed=99712,episodes=2,max_episode_ticks=41,policy_kind=kind,
                 action_seed=99713 if kind=='random' else None)
    full=FreshPongEvaluator(None,PongEvaluationConfig(**options))
    fast=FreshPongEvaluator(None,PongEvaluationConfig(**options,skip_baseline_perception=True))
    full_rows=[];fast_rows=[]
    while not full.complete:full_rows.append(full.step())
    while not fast.complete:fast_rows.append(fast.step())
    # Perception diagnostics are absent on the explicitly skipped path.
    assert all(row['input_block_norm2'] is not None for row in full_rows)
    assert all(row['input_block_norm2'] is None for row in fast_rows)
    assert all(row['policy_block_norm2'] is not None for row in full_rows)
    assert all(row['policy_block_norm2'] is None for row in fast_rows)
    diagnostics={'input_block_norm2','policy_block_norm2'}
    assert [{k:v for k,v in row.items() if k not in diagnostics} for row in full_rows] == [
        {k:v for k,v in row.items() if k not in diagnostics} for row in fast_rows]
    assert torch.equal(full.env.game.rng,fast.env.game.rng)
    assert torch.equal(full.env.game.ball,fast.env.game.ball)
    assert torch.equal(full.env.game.body.position,fast.env.game.body.position)
    assert full.physics_seconds == fast.physics_seconds
    assert full.initial_cameras == fast.initial_cameras == 2


def test_evaluation_detects_frozen_parameter_changes():
    torch.set_num_threads(1)
    evaluator=FreshPongEvaluator(None,PongEvaluationConfig(99714,policy_kind='initial'))
    evaluator.vision.observer.weights.numpy()[0,0]+=1
    with pytest.raises(RuntimeError,match='changed frozen'):
        evaluator.step()


def test_durable_evaluation_replays_mid_rally_and_terminal_boundary(tmp_path):
    torch.set_num_threads(1)
    config=PongEvaluationConfig(99715,episodes=3,max_episode_ticks=4,policy_kind='random',action_seed=99716)
    run=DurableRun(tmp_path/'eval.db',session=FreshPongEvaluator(None,config),snapshot_transitions=3)
    expected=[run.step() for _ in range(5)]
    end=fingerprint(run.session);run.close_without_checkpoint()
    restored=DurableRun(tmp_path/'eval.db')
    assert restored.replayed==2 and fingerprint(restored.session)==end
    assert restored.session.episodes==2
    restored.step()
    assert restored.session.complete
    assert restored.session.summary()['truncations']==3
    restored.stop()


def test_blank_camera_ablation_masks_only_policy_observations():
    torch.set_num_threads(1)
    dimension=VisualStateEncoder(64,64,.02).dimension
    weights=np.full((3,dimension),.001);weights[2]*=2
    config=PongEvaluationConfig(99717,episodes=2,max_episode_ticks=11,blank_camera=True)
    evaluator=FreshPongEvaluator(weights,config)
    while not evaluator.complete:
        row=evaluator.step()
        assert not evaluator.previous.any()
        assert not evaluator.latest_encoded['entities'] and not evaluator.latest_encoded['forecasts']
    assert evaluator.env.game.render(64,64).any()  # Physics/camera acquisition still has a scene.
