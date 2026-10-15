import pickle
import random
import sqlite3

import numpy as np
import pytest
import torch
from fly_connectome.control.codec import FrozenCodec, fingerprint, frozen_tensors
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.session import LearningSession


@pytest.fixture
def session():
    torch.set_num_threads(1)
    random.seed(33); np.random.seed(33); torch.manual_seed(33)
    return LearningSession(max_episode_ticks=9)


def test_dedup_preserves_aliases_rng_and_complete_state(session):
    session.vision.state.memory._variance_keys = session.vision.state.memory.keys
    session.vision.state.memory._variance_version = session.vision.state.memory.keys._version
    codec = FrozenCodec.from_session(session)
    payload = codec.dumps(session)
    restored = codec.loads(payload)
    assert fingerprint(restored) == fingerprint(session)
    assert restored.vision.state.memory._variance_keys is restored.vision.state.memory.keys
    assert restored.body_alias is restored.env.game.body
    assert restored.learner.rng.bit_generator.state == session.learner.rng.bit_generator.state
    assert len(payload) < len(pickle.dumps(session,protocol=5))
    restored.vision.observer.weights.add_(1)
    with pytest.raises(ValueError,match='changed'):
        codec.dumps(restored)


def test_tail_replay_no_missing_or_duplicate_commits(tmp_path,session):
    path = tmp_path/'run.db'
    run = DurableRun(path,session=session,snapshot_transitions=3)
    expected = [run.step() for _ in range(5)]
    final = fingerprint(run.session)
    assert run.db.execute('SELECT count FROM replay_active').fetchone()[0] == 3
    run.close_without_checkpoint()
    resumed = DurableRun(path)
    assert resumed.replayed == 2
    assert fingerprint(resumed.session) == final
    assert resumed.session.transitions == 5
    assert resumed.db.execute('SELECT count(*) FROM replay_samples').fetchone()[0] == 5
    assert resumed.db.execute('SELECT count(*) FROM replay_artifacts').fetchone()[0] == 1
    resumed.stop()
    stopped = DurableRun(path)
    assert stopped.replayed == 0
    assert stopped.session.transitions == 5
    stopped.stop()


def test_interval_seconds_and_stop_checkpoint(tmp_path,session):
    class Clock:
        now = 0
        def time_ns(self): return self.now
        def monotonic_ns(self): return self.now
    clock = Clock()
    run = DurableRun(tmp_path/'run.db',session=session,clock=clock)
    run.step()
    assert run.db.execute('SELECT count FROM replay_active').fetchone()[0] == 0
    clock.now = 10_000_000_000
    run.step()
    assert run.db.execute('SELECT count FROM replay_active').fetchone()[0] == 2
    run.stop()


def test_failed_snapshot_rolls_back_sample_and_poisoned_writer(tmp_path,session):
    path = tmp_path/'run.db'
    run = DurableRun(path,session=session,snapshot_transitions=1)
    run.db.execute("CREATE TRIGGER fail BEFORE INSERT ON replay_active BEGIN SELECT RAISE(ABORT,'injected'); END")
    with pytest.raises(sqlite3.IntegrityError): run.step()
    assert run.db.execute('SELECT count(*) FROM replay_samples').fetchone()[0] == 0
    with pytest.raises(RuntimeError): run.step()
    run.close_without_checkpoint()
    recovered = DurableRun(path)
    assert recovered.session.transitions == 0
    recovered.db.execute('DROP TRIGGER fail')
    recovered.step(); recovered.stop()


def test_corrupt_tail_fails_closed(tmp_path,session):
    path=tmp_path/'run.db'
    run=DurableRun(path,session=session)
    run.step()
    run.db.execute("UPDATE replay_samples SET state_hash='wrong'")
    run.close_without_checkpoint()
    with pytest.raises(ValueError,match='hash mismatch'):
        DurableRun(path)


def test_restart_gap_is_distinct_and_persisted(tmp_path,session):
    class Clock:
        now = 1_000_000_000
        def time_ns(self): return self.now
        def monotonic_ns(self): return self.now
    clock = Clock()
    path=tmp_path/'run.db'
    run=DurableRun(path,session=session,clock=clock)
    clock.now += 2_000_000_000
    run.step()
    run.stop()
    clock.now += 7_000_000_000
    resumed=DurableRun(path,clock=clock)
    summary=resumed.timing_summary()
    assert summary[0]['accounted_process_ns']==2_000_000_000
    assert summary[0]['clean_stop']
    assert summary[1]['unobserved_gap_ns']==7_000_000_000
    resumed.stop()


def test_committed_tail_replay_mismatch_rejected_even_with_valid_hash_chain(tmp_path,session):
    from fly_connectome.control.journal import canonical,digest
    path=tmp_path/'run.db'
    run=DurableRun(path,session=session)
    record=run.step()
    record['reward']=123.
    raw=canonical(record)
    state_hash=run.db.execute('SELECT state_hash FROM replay_samples').fetchone()[0]
    chain=digest(canonical(['',0,raw.decode(),state_hash]))
    run.db.execute('UPDATE replay_samples SET record=?,chain_hash=?',(raw,chain))
    run.close_without_checkpoint()
    with pytest.raises(ValueError,match='replay mismatch'):
        DurableRun(path)


def test_runtime_migration_is_not_silently_accepted(tmp_path,session):
    import json
    path=tmp_path/'run.db'
    run=DurableRun(path,session=session)
    config=json.loads(run.db.execute('SELECT config FROM replay_meta').fetchone()[0])
    config['runtime']['torch']='other-runtime'
    run.db.execute('UPDATE replay_meta SET config=?',(json.dumps(config).encode(),))
    run.close_without_checkpoint()
    with pytest.raises(ValueError,match='incompatible runtime'):
        DurableRun(path)
