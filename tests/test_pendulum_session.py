"""Deferred native-survival / mutable-vision integration and source bridge checks."""
import hashlib
import pickle
from pathlib import Path
import numpy as np
import pytest
import torch
from fly_connectome.control.codec import fingerprint,frozen_tensors
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.online_vision import VisionSnapshot
from fly_connectome.control.pendulum_session import PendulumLearningSession
from fly_connectome.control.session import source_signature
from fly_connectome.control.vision_transfer import bridge_snapshot


def test_native_survival_learning_recovery_and_eager_initial_image_count(tmp_path):
    torch.set_num_threads(1)
    reference=PendulumLearningSession(739397,max_episode_ticks=25)
    run=DurableRun(tmp_path/'balance.db',session=PendulumLearningSession(739397,max_episode_ticks=25),
                   snapshot_transitions=5)
    assert set(frozen_tensors(run.session))=={'observer.weights','observer.bias'}
    for _ in range(12):
        record=run.step();assert record==reference.step()
        assert record['reward']==record['elapsed_seconds'] and record['hits']==0
        if record['truncated']:
            assert not record['terminated'] and record['outgoing_discount']>0 and record['bootstrap_action'] is not None
            assert record['visual_learning']['reset_initial_image_added']
    assert run.session.episodes==1 and run.session.visual_learning_observations==14
    assert np.any(run.session.learner.weights)
    expected=fingerprint(run.session);run.close_without_checkpoint()
    restored=DurableRun(tmp_path/'balance.db')
    assert restored.replayed==2 and fingerprint(restored.session)==expected
    for _ in range(5):assert restored.step()==reference.step()
    assert fingerprint(restored.session)==fingerprint(reference)
    snapshot=restored.session.retained_vision()
    assert snapshot.learning_observations==19 and not snapshot.vision.pending
    assert snapshot.vision.histories=={} and snapshot.vision.state.keys=={}
    restored.stop()


def test_control_source_bridge_is_explicit_owned_and_rejects_visual_changes(tmp_path):
    torch.set_num_threads(1)
    session=PendulumLearningSession(739397,max_episode_ticks=25)
    for _ in range(12):session.step()
    actual=session.retained_vision();current=source_signature()
    # Self-contained source fixture: one reviewed control-only file deletion.
    ancestor=dict(current,**{'control/retired_control_fixture.py':'a'*64})
    old=VisionSnapshot(actual.vision,actual.content_hash,ancestor,actual.prior_model_hash,
                       actual.vision_configuration,actual.learning_observations)
    path=tmp_path/'ancestor.pkl';path.write_bytes(pickle.dumps(old,protocol=5))
    file_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    changes={'control/retired_control_fixture.py':dict(before='a'*64,after=None)}
    moved,receipt=bridge_snapshot(path,expected_file_sha256=file_hash,
        ancestor_source=ancestor,reviewed_control_changes=changes)
    assert moved.validate().content_hash==actual.content_hash
    assert moved.learning_observations==actual.learning_observations==14
    assert receipt['observations_added']==0 and receipt['visual_code_unchanged']
    assert moved.vision.dynamics[1].short.keys.data_ptr()!=actual.vision.dynamics[1].short.keys.data_ptr()
    untouched=fingerprint(actual.vision)
    moved.vision.dynamics[1].short.seen+=1
    assert fingerprint(actual.vision)==untouched
    with pytest.raises(ValueError,match='bytes changed'):
        bridge_snapshot(path,expected_file_sha256='0'*64,ancestor_source=ancestor,reviewed_control_changes=changes)
    with pytest.raises(ValueError,match='control-only'):
        bridge_snapshot(path,expected_file_sha256=file_hash,ancestor_source=ancestor,reviewed_control_changes={})
    changed_ancestor=dict(ancestor);changed_ancestor['vision/state.py']='b'*64
    visual_changes=dict(changes,**{'vision/state.py':dict(before='b'*64,after=current['vision/state.py'])})
    with pytest.raises(ValueError,match='control-only'):
        bridge_snapshot(path,expected_file_sha256=file_hash,ancestor_source=changed_ancestor,
            reviewed_control_changes=visual_changes)


def test_continuing_survival_rejects_undiscounted_values():
    with pytest.raises(ValueError,match='discount below one'):
        PendulumLearningSession(gamma_per_second=1.)


def test_environment_and_exploration_have_independent_pinned_seed_streams():
    torch.set_num_threads(1)
    first=PendulumLearningSession(739397,learner_seed=739409)
    second=PendulumLearningSession(739397,learner_seed=739411)
    assert fingerprint(first.env)==fingerprint(second.env)
    assert first.config['learner_seed']==739409 and second.config['learner_seed']==739411
    assert fingerprint(first.learner.rng)!=fingerprint(second.learner.rng)
    default=PendulumLearningSession(739397)
    assert default.config['learner_seed']==int(np.random.SeedSequence([739397,824637]).generate_state(1)[0])
    assert default.config['learner_seed']!=default.config['seed']
