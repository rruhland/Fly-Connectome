"""FULL SQLite sample journal with interval snapshots and verified deterministic replay.

Only committed transitions exist on recovery. A failed in-memory writer is poisoned;
reopen from the journal instead of attempting to continue its advanced runtime.
"""
import hashlib
import json
import sqlite3
import time
import platform
import sys
import numpy as np
import torch

from .codec import FrozenCodec, fingerprint
from .journal import canonical, digest
from .session import LearningSession


def runtime_signature():
    return dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
                platform=sys.platform,machine=platform.machine())


class DurableRun:
    def __init__(self, path, *, session=None, snapshot_transitions=100,
                 snapshot_seconds=10., clock=time):
        if (type(snapshot_transitions) is not int or snapshot_transitions < 1
                or not 0 < snapshot_seconds < float('inf')):
            raise ValueError('positive snapshot intervals required')
        self.clock, self.started_wall, self.started_mono = clock, clock.time_ns(), clock.monotonic_ns()
        self.db = sqlite3.connect(path,timeout=30,isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS replay_meta (id INTEGER PRIMARY KEY CHECK(id=1), config BLOB NOT NULL);
          CREATE TABLE IF NOT EXISTS replay_artifacts (hash TEXT PRIMARY KEY, payload BLOB NOT NULL);
          CREATE TABLE IF NOT EXISTS replay_samples (seq INTEGER PRIMARY KEY, record BLOB NOT NULL,
              state_hash TEXT NOT NULL, chain_hash TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS replay_active (id INTEGER PRIMARY KEY CHECK(id=1),
              count INTEGER NOT NULL, payload BLOB NOT NULL, payload_hash TEXT NOT NULL, state_hash TEXT NOT NULL, wall_ns INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS replay_timing (id INTEGER PRIMARY KEY, kind TEXT NOT NULL,
              count INTEGER NOT NULL, wall_ns INTEGER NOT NULL, process_ns INTEGER NOT NULL, gap_ns INTEGER);
        ''')
        self.failed = self.stopped = False
        config = self.db.execute('SELECT config FROM replay_meta WHERE id=1').fetchone()
        if config is None:
            if session is None:
                self.db.close()
                raise ValueError('new journal needs an initial session')
            if session.transitions != 0:
                self.db.close()
                raise ValueError('initial session must precede first transition')
            self.session = session
            self.codec = FrozenCodec.from_session(session)
            config = dict(schema='m1b-replay-v2',bank=self.codec.key,runtime=runtime_signature(),
                          session_kind=session.durable_kind,
                          snapshot_transitions=snapshot_transitions,snapshot_seconds=snapshot_seconds)
            self.db.execute('BEGIN IMMEDIATE')
            try:
                self.db.execute('INSERT INTO replay_meta VALUES (1,?)',(canonical(config),))
                self.db.execute('INSERT INTO replay_artifacts VALUES (?,?)',(self.codec.key,self.codec.artifact))
                self._checkpoint()
                self.db.execute('COMMIT')
            except BaseException:
                self.db.execute('ROLLBACK')
                self.db.close()
                raise
            self.last_snapshot_count = self.replayed = 0
        else:
            if session is not None:
                self.db.close()
                raise ValueError('existing journal must be resumed, not overwritten')
            config = json.loads(config[0])
            if config['schema'] != 'm1b-replay-v2':
                self.db.close()
                raise ValueError('unsupported replay schema')
            if config.get('runtime') != runtime_signature():
                self.db.close()
                raise ValueError('incompatible runtime; migration requires a separate proof')
            self._recover(config['bank'],config['session_kind'])
        self.snapshot_transitions, self.snapshot_seconds = config['snapshot_transitions'],config['snapshot_seconds']
        self.last_snapshot_mono = self.clock.monotonic_ns()-max(0,self.clock.time_ns()-self.last_snapshot_wall)
        last = self.db.execute('SELECT wall_ns FROM replay_timing ORDER BY id DESC LIMIT 1').fetchone()
        gap = None if last is None else self.started_wall-last[0]
        self._timing('start',gap)

    def _timing(self, kind, gap=None):
        self.db.execute('INSERT INTO replay_timing(kind,count,wall_ns,process_ns,gap_ns) VALUES (?,?,?,?,?)',
                        (kind,self.session.transitions,self.started_wall if kind == 'start' else self.clock.time_ns(),
                         self.clock.monotonic_ns()-self.started_mono,gap))

    def _checkpoint(self):
        self.session.capture_rng()
        payload = self.codec.dumps(self.session)
        self.last_snapshot_wall = self.clock.time_ns()
        self.db.execute('INSERT OR REPLACE INTO replay_active VALUES (1,?,?,?,?,?)',
                        (self.session.transitions,payload,digest(payload),fingerprint(self.session),self.last_snapshot_wall))

    def _recover(self, bank_key, session_kind):
        # One consistent read transaction. CAS on append rejects competing writers.
        self.db.execute('BEGIN')
        try:
            bank = self.db.execute('SELECT payload FROM replay_artifacts WHERE hash=?',(bank_key,)).fetchone()
            if bank is None or digest(bank[0]) != bank_key:
                raise ValueError('corrupt or missing frozen bank')
            self.codec = FrozenCodec(bank[0])
            previous, records = '', []
            for seq,raw,state_hash,actual in self.db.execute('SELECT * FROM replay_samples ORDER BY seq'):
                expected = digest(canonical([previous,seq,raw.decode(),state_hash]))
                if seq != len(records) or actual != expected:
                    raise ValueError('sample sequence/hash mismatch')
                records.append((json.loads(raw),state_hash))
                previous = actual
            active = self.db.execute('SELECT count,payload,payload_hash,state_hash,wall_ns FROM replay_active WHERE id=1').fetchone()
            if active is None or not 0 <= active[0] <= len(records) or digest(active[1]) != active[2]:
                raise ValueError('corrupt active checkpoint')
            count,payload,_,state_hash,self.last_snapshot_wall = active
            if count and state_hash != records[count-1][1]:
                raise ValueError('checkpoint/sample state mismatch')
            from .toy_session import ToySession
            classes = {'pong-learning-v1':LearningSession,'toy-session-v1':ToySession}
            if session_kind not in classes:
                raise ValueError('unsupported session kind')
            self.session = classes[session_kind].validate_restored(self.codec.loads(payload))
            if self.session.transitions != count or fingerprint(self.session) != state_hash:
                raise ValueError('restored state mismatch')
            self.last_snapshot_count = count
            self.replayed = len(records)-count
            for expected_record,expected_hash in records[count:]:
                actual_record = self.session.step()
                if actual_record != expected_record or fingerprint(self.session) != expected_hash:
                    raise ValueError('deterministic replay mismatch')
        finally:
            self.db.execute('ROLLBACK')

    def step(self):
        if self.failed or self.stopped:
            raise RuntimeError('writer unavailable; reopen journal')
        self.failed = True
        expected_count = self.session.transitions
        record = self.session.step()
        raw, state_hash = canonical(record),fingerprint(self.session)
        due = (self.session.transitions-self.last_snapshot_count >= self.snapshot_transitions
               or self.clock.monotonic_ns()-self.last_snapshot_mono >= self.snapshot_seconds*1e9)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            head = self.db.execute('SELECT seq,chain_hash FROM replay_samples ORDER BY seq DESC LIMIT 1').fetchone()
            count,previous = (0,'') if head is None else (head[0]+1,head[1])
            if expected_count != count:
                raise ValueError('stale writer sequence')
            chain = digest(canonical([previous,count,raw.decode(),state_hash]))
            self.db.execute('INSERT INTO replay_samples VALUES (?,?,?,?)',(count,raw,state_hash,chain))
            if due:
                self._checkpoint()
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise
        if due:
            self.last_snapshot_count = self.session.transitions
            self.last_snapshot_mono = self.clock.monotonic_ns()
        self._timing('heartbeat')
        self.failed = False
        return record

    def stop(self):
        if self.failed or self.stopped:
            raise RuntimeError('writer unavailable; reopen journal')
        self.failed = True
        self.db.execute('BEGIN IMMEDIATE')
        try:
            count = self.db.execute('SELECT count(*) FROM replay_samples').fetchone()[0]
            if count != self.session.transitions:
                raise ValueError('stale stop checkpoint')
            self._checkpoint()
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise
        self._timing('stop')
        self.stopped = True
        self.db.close()

    def close_without_checkpoint(self):
        """For failed/cancelled processes: preserve the last durable sample prefix."""
        self.stopped = True
        self.db.close()

    def timing_summary(self):
        """Account completed process prefixes; never claim unseen crash tails are zero."""
        segments = []
        for kind,count,wall,cost,gap in self.db.execute(
                'SELECT kind,count,wall_ns,process_ns,gap_ns FROM replay_timing ORDER BY id'):
            if kind == 'start':
                segments.append(dict(start_wall_ns=wall,accounted_process_ns=cost,
                                     unobserved_gap_ns=gap,clean_stop=False,last_committed_count=count))
            else:
                if not segments:
                    raise ValueError('timing event without process start')
                segments[-1].update(accounted_process_ns=cost,clean_stop=kind=='stop',
                                    last_committed_count=count)
        return segments
