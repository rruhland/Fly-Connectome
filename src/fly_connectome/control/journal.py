"""Atomic sample + active continuation journal (local trusted state bytes).

SQLite FULL transactions commit sample and continuation together. Only the active
state is retained; audit samples form a SHA256 chain. Digests detect corruption,
not malicious edits. State serialization belongs to the runtime, not this store.
"""

import hashlib
import json
import sqlite3


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def chain(previous, seq, sample, state_hash):
    return digest(canonical([previous, seq, sample.decode(), state_hash]))


class Journal:
    def __init__(self, path):
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS artifacts (hash TEXT PRIMARY KEY, payload BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS samples (
                seq INTEGER PRIMARY KEY, sample BLOB NOT NULL,
                state_hash TEXT NOT NULL, chain_hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS active (
                id INTEGER PRIMARY KEY CHECK(id=1), seq INTEGER NOT NULL,
                payload BLOB NOT NULL, state_hash TEXT NOT NULL);
        ''')
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('schema', 'm1b-reconstruction-v1')")
        if self.db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()[0] != 'm1b-reconstruction-v1':
            self.close()
            raise ValueError('unsupported journal schema')

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def put_artifact(self, payload):
        key = digest(payload)
        self.db.execute('INSERT OR IGNORE INTO artifacts VALUES (?, ?)', (key, payload))
        if self.get_artifact(key) != payload:
            raise ValueError('artifact content mismatch')
        return key

    def get_artifact(self, key):
        row = self.db.execute('SELECT payload FROM artifacts WHERE hash=?', (key,)).fetchone()
        if row is None or digest(row[0]) != key:
            raise ValueError('missing or corrupt artifact')
        return row[0]

    def commit(self, expected_seq, sample, state):
        """CAS append: first sequence is zero; stale writers cannot overwrite."""
        if type(expected_seq) is not int or expected_seq < 0 or not isinstance(state, bytes):
            raise ValueError('nonnegative sequence and serialized bytes required')
        sample_bytes = canonical(sample)
        state_hash = digest(state)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            head = self.db.execute('SELECT seq, chain_hash FROM samples ORDER BY seq DESC LIMIT 1').fetchone()
            next_seq, previous = (0, '') if head is None else (head[0]+1, head[1])
            if expected_seq != next_seq:
                raise ValueError('stale or noncontiguous sequence')
            self.db.execute('INSERT INTO samples VALUES (?, ?, ?, ?)',
                            (next_seq, sample_bytes, state_hash,
                             chain(previous, next_seq, sample_bytes, state_hash)))
            self.db.execute('INSERT OR REPLACE INTO active VALUES (1, ?, ?, ?)',
                            (next_seq, state, state_hash))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def load(self):
        """Verify the entire audit chain before returning active state bytes."""
        self.db.execute('BEGIN')
        try:
            previous, count, last_state_hash = '', 0, None
            for seq, sample, state_hash, actual in self.db.execute('SELECT * FROM samples ORDER BY seq'):
                if seq != count or actual != chain(previous, seq, sample, state_hash):
                    raise ValueError('corrupt sample sequence or hash chain')
                json.loads(sample)
                previous, count, last_state_hash = actual, count+1, state_hash
            active = self.db.execute('SELECT seq, payload, state_hash FROM active WHERE id=1').fetchone()
            if not count:
                if active is not None:
                    raise ValueError('orphan active checkpoint')
                return None
            if (active is None or active[0] != count-1 or active[2] != last_state_hash
                    or digest(active[1]) != active[2]):
                raise ValueError('corrupt active checkpoint')
            return active[0], active[1]
        finally:
            self.db.execute('ROLLBACK')

    def samples(self):
        self.load()
        return [json.loads(row[0]) for row in self.db.execute('SELECT sample FROM samples ORDER BY seq')]
