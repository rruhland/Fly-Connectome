"""Deferred optional continuation checks; original defaults remain plastic."""
import hashlib
import numpy as np
import pytest
import torch
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.pendulum_session import PendulumLearningSession
from fly_connectome.vision import VisualStateEncoder


def test_owned_weights_are_installed_before_initial_pending_action():
    torch.set_num_threads(1);encoder=VisualStateEncoder(64,64,.02)
    weights=np.zeros((3,encoder.dimension));weights[2,encoder.layout['metadata'][0]+3]=1.
    original=weights.copy()
    s=PendulumLearningSession(104729,learner_seed=104743,weights=weights,epsilon=0.)
    assert s.learner.pending_action==2 and np.array_equal(s.learner.weights,original)
    assert s.learner.transitions==0 and not np.any(s.learner.traces)
    assert s.learner.q_old==s.learner.incoming_discount==0.
    assert s.config['starting_learner_weights_sha256']==hashlib.sha256(original.tobytes()).hexdigest()
    weights[:]=99.;assert np.array_equal(s.learner.weights,original)
    assert s.step()['action']==2


def test_explicit_original_defaults_preserve_graph_and_records():
    torch.set_num_threads(1)
    first=PendulumLearningSession(104729,learner_seed=104743,max_episode_ticks=25)
    second=PendulumLearningSession(104729,learner_seed=104743,max_episode_ticks=25,
        weights=None,learn_enabled=True)
    assert not hasattr(first,'learn_enabled') and not hasattr(first,'frozen_visual_hash')
    assert fingerprint(first)==fingerprint(second)
    for _ in range(12):assert first.step()==second.step()
    assert fingerprint(first)==fingerprint(second)


@pytest.mark.parametrize('enabled',[True,False])
def test_restart_weights_memory_ownership_and_recovery_across_eager_reset(tmp_path,enabled):
    torch.set_num_threads(1)
    ancestor=PendulumLearningSession(104759,learner_seed=104761,max_episode_ticks=25)
    for _ in range(12):ancestor.step()
    snapshot=ancestor.retained_vision();before=fingerprint(snapshot.vision)
    weights=ancestor.learner.weights.copy()
    args=dict(seed=104773,learner_seed=104779,max_episode_ticks=25,
        vision_snapshot=snapshot,weights=weights,learn_enabled=enabled)
    reference=PendulumLearningSession(**args)
    run=DurableRun(tmp_path/'continuation.db',session=PendulumLearningSession(**args),snapshot_transitions=5)
    assert run.session.transitions==run.session.learner.transitions==0
    assert run.session.starting_visual_learning_observations==snapshot.learning_observations
    for _ in range(12):assert run.step()==reference.step()
    expected=fingerprint(run.session);run.close_without_checkpoint()
    restored=DurableRun(tmp_path/'continuation.db')
    assert restored.replayed==2 and fingerprint(restored.session)==expected
    for _ in range(5):assert restored.step()==reference.step()
    assert fingerprint(restored.session)==fingerprint(reference)
    assert fingerprint(snapshot.vision)==before
    s=restored.session;assert s.episodes==1 and s.visual_observations==19
    assert s.visual_learning_observations==(19 if enabled else 0)
    result=s.retained_vision()
    assert result.learning_observations==snapshot.learning_observations+(19 if enabled else 0)
    if not enabled:assert s._frozen_visual_hash()==s.frozen_visual_hash
    assert not np.array_equal(s.learner.weights,weights)
    restored.stop()


def test_off_memory_mutation_fails_before_camera_commit_and_poisoned_writer(tmp_path):
    torch.set_num_threads(1)
    run=DurableRun(tmp_path/'frozen.db',session=PendulumLearningSession(104729,learner_seed=104743,learn_enabled=False))
    original=fingerprint(run.session)
    run.session.vision.dynamics[1].short.seen+=1
    with pytest.raises(RuntimeError,match='learned visual state'):run.step()
    assert run.failed and run.db.execute('SELECT count(*) FROM replay_samples').fetchone()[0]==0
    with pytest.raises(RuntimeError):run.step()
    run.close_without_checkpoint();restored=DurableRun(tmp_path/'frozen.db')
    assert fingerprint(restored.session)==original and restored.replayed==0
    restored.close_without_checkpoint()


def test_visual_mode_is_not_truthy_coercion():
    with pytest.raises(ValueError,match='bool visual'):PendulumLearningSession(learn_enabled=1)
