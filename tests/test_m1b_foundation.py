import sqlite3

import numpy as np
import pytest

from fly_connectome.control import FrozenLinearPolicy, Journal


def test_frozen_nonzero_values_legal_actions_and_no_mutation():
    weights = np.array([[1., 2., 3.], [-4., 5., 6.]])
    original = weights.copy()
    policy = FrozenLinearPolicy(weights)
    weights[:] = 0
    for _ in range(5):
        action, q = policy.choose([0, 2], [2., -1.])
        np.testing.assert_array_equal(q, [-1., -14.])
        assert action == 0
        assert policy.choose([0, 2], [2., -1.], [1])[0] == 1
    np.testing.assert_array_equal(policy.weights, original)
    policy.weights[:] = 99
    np.testing.assert_array_equal(policy.weights, original)


@pytest.mark.parametrize('indices,values', [([0, 0], [1, 2]), ([-1], [1]),
    ([3], [1]), ([.5], [1]), ([0], [float('nan')]), ([0], [])])
def test_invalid_features(indices, values):
    with pytest.raises(ValueError):
        FrozenLinearPolicy(np.ones((2, 3))).q_values(indices, values)


def test_journal_reopen_sequence_and_artifact(tmp_path):
    path = tmp_path/'run.db'
    with Journal(path) as journal:
        assert journal.db.execute('PRAGMA synchronous').fetchone()[0] == 2
        assert journal.load() is None
        key = journal.put_artifact(b'model')
        assert journal.put_artifact(b'model') == key
        journal.commit(0, {'reward': 1}, b'first')
        with pytest.raises(ValueError, match='sequence'):
            journal.commit(0, {}, b'stale')
        journal.commit(1, {'reward': -1}, b'second')
    with Journal(path) as journal:
        assert journal.load() == (1, b'second')
        assert journal.samples() == [{'reward': 1}, {'reward': -1}]
        assert journal.get_artifact(key) == b'model'
        assert journal.db.execute('SELECT count(*) FROM active').fetchone()[0] == 1
        assert journal.db.execute('SELECT count(*) FROM artifacts').fetchone()[0] == 1


def test_atomic_rollback_between_sample_and_checkpoint(tmp_path):
    with Journal(tmp_path/'run.db') as journal:
        journal.commit(0, {}, b'old')
        journal.db.execute("CREATE TRIGGER injected BEFORE INSERT ON active BEGIN SELECT RAISE(ABORT, 'injected'); END")
        with pytest.raises(sqlite3.IntegrityError):
            journal.commit(1, {}, b'new')
        assert journal.load() == (0, b'old')
        assert len(journal.samples()) == 1


@pytest.mark.parametrize('sql', [
    "UPDATE active SET payload=x'00'",
    "UPDATE samples SET sample=x'7b7d' WHERE seq=0",
    'DELETE FROM samples WHERE seq=0',
    'DELETE FROM active',
])
def test_corruption_rejected(tmp_path, sql):
    with Journal(tmp_path/'run.db') as journal:
        journal.commit(0, {'reward': 1}, b'old')
        journal.commit(1, {'reward': 2}, b'new')
        journal.db.execute(sql)
        with pytest.raises(ValueError, match='corrupt'):
            journal.load()


def test_two_writers_cannot_replace_committed_continuation(tmp_path):
    path = tmp_path/'run.db'
    with Journal(path) as first, Journal(path) as second:
        first.commit(0, {}, b'one')
        with pytest.raises(ValueError):
            second.commit(0, {}, b'two')
        assert second.load() == (0, b'one')


def test_process_exit_during_transaction_preserves_last_commit(tmp_path):
    import subprocess
    import sys
    path = tmp_path/'run.db'
    with Journal(path) as journal:
        journal.commit(0, {'reward': 1}, b'old')
    code = '''
import os, sqlite3, sys
connection = sqlite3.connect(sys.argv[1], isolation_level=None)
connection.execute('PRAGMA synchronous=FULL')
connection.execute('BEGIN IMMEDIATE')
connection.execute("UPDATE active SET payload=x'00'")
os._exit(19)
'''
    result = subprocess.run([sys.executable, '-c', code, str(path)], timeout=10)
    assert result.returncode == 19
    with Journal(path) as journal:
        assert journal.load() == (0, b'old')
        assert journal.samples() == [{'reward': 1}]


def test_artifact_corruption_rejected(tmp_path):
    with Journal(tmp_path/'run.db') as journal:
        key = journal.put_artifact(b'model')
        journal.db.execute("UPDATE artifacts SET payload=x'00'")
        with pytest.raises(ValueError, match='corrupt artifact'):
            journal.get_artifact(key)
