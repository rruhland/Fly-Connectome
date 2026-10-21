"""Deferred genuinely frozen held-out task evaluation and simple controls."""
import random
from dataclasses import asdict
import numpy as np
import pytest
import torch
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.online_vision import learned_visual_state
from fly_connectome.control.pendulum_evaluation import FreshPendulumEvaluator,PendulumEvaluationConfig
from fly_connectome.control.pendulum_session import PendulumLearningSession


def test_fresh_probe_cannot_update_policy_vision_or_ancestor_and_recovers_exactly(tmp_path):
    torch.set_num_threads(1)
    training=PendulumLearningSession(739397,max_episode_ticks=25)
    for _ in range(12):training.step()
    snapshot=training.retained_vision();snapshot_hash=fingerprint(snapshot)
    weights,cap=training.mapped_evaluation()
    cfg=PendulumEvaluationConfig(739399,max_episode_ticks=25,policy_scaling=cap)
    outside=fingerprint((random.getstate(),np.random.get_state(),torch.get_rng_state()))
    reference=FreshPendulumEvaluator(weights,cfg,vision_snapshot=snapshot)
    run=DurableRun(tmp_path/'frozen.db',session=FreshPendulumEvaluator(weights,cfg,vision_snapshot=snapshot),
                   snapshot_transitions=5)
    learned=fingerprint(learned_visual_state(run.session.vision))
    for _ in range(7):assert run.step()==reference.step()
    state=fingerprint(run.session);run.close_without_checkpoint()
    restored=DurableRun(tmp_path/'frozen.db')
    assert restored.replayed==2 and fingerprint(restored.session)==state
    while not reference.complete:assert restored.step()==reference.step()
    assert np.array_equal(restored.session.policy.weights,weights)
    assert fingerprint(learned_visual_state(restored.session.vision))==learned
    assert fingerprint(snapshot)==snapshot_hash
    assert restored.session.retained_vision().learning_observations==snapshot.learning_observations
    assert outside==fingerprint((random.getstate(),np.random.get_state(),torch.get_rng_state()))
    restored.stop()


@pytest.mark.parametrize('kind,hold',[('negative',1),('noop',1),('positive',1),('random',1),('random',10)])
def test_baseline_perception_skip_preserves_all_actions_physics_and_rewards(kind,hold):
    torch.set_num_threads(1)
    options=dict(seed=739399,max_episode_ticks=25,policy_kind=kind,action_seed=739403,action_hold_frames=hold)
    observed=FreshPendulumEvaluator(None,PendulumEvaluationConfig(**options))
    skipped=FreshPendulumEvaluator(None,PendulumEvaluationConfig(**options,skip_baseline_perception=True))
    while not observed.complete:
        expected=observed.step();actual=skipped.step()
        expected.pop('visual_observations');actual.pop('visual_observations')
        assert actual==expected and fingerprint(observed.env)==fingerprint(skipped.env)
    assert skipped.visual_observations==0 and observed.visual_observations==observed.transitions+1


def test_json_configuration_restores_typed_physics():
    config=PendulumEvaluationConfig(739399,policy_kind='random',action_seed=739403,action_hold_frames=10)
    assert PendulumEvaluationConfig.from_dict(asdict(config))==config
