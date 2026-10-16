"""Isolated worker checkpoint contract; no candidate source changes."""
import importlib.util
from pathlib import Path
import sqlite3
import pytest
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.toy_session import ToySession,ToySpec

path=Path(__file__).parents[1]/'scripts/experimental/run_pong_development.py'
spec=importlib.util.spec_from_file_location('warm_worker',path)
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


def test_chunk_checkpoint_retains_writer_and_full_recovery(tmp_path):
    journal=tmp_path/'run.db';run=DurableRun(journal,session=ToySession(ToySpec("delayed",991827,4,diagnostic=True)),snapshot_transitions=100)
    for _ in range(3):run.step()
    worker.checkpoint_open_run(run)
    assert not run.failed and not run.stopped
    assert run.last_snapshot_count==3
    assert run.db.execute('SELECT count FROM replay_active').fetchone()[0]==3
    expected=run.step()
    run.close_without_checkpoint()
    recovered=DurableRun(journal)
    assert recovered.replayed==1 and recovered.session.transitions==4
    assert expected['transition']==3
    recovered.stop()


def test_stale_chunk_checkpoint_poisons_writer(tmp_path):
    journal=tmp_path/'run.db';run=DurableRun(journal,session=ToySession(ToySpec("delayed",991827,4,diagnostic=True)))
    other=DurableRun(journal)
    other.step()
    with pytest.raises(ValueError,match='stale chunk'):worker.checkpoint_open_run(run)
    assert run.failed
    with pytest.raises(RuntimeError):run.step()
    run.close_without_checkpoint();other.stop()


def test_failed_chunk_checkpoint_preserves_last_committed_prefix(tmp_path,monkeypatch):
    journal=tmp_path/'run.db';run=DurableRun(journal,session=ToySession(ToySpec("delayed",991827,4,diagnostic=True)))
    run.step()
    def fail():raise OSError('injected snapshot failure')
    monkeypatch.setattr(run,'_checkpoint',fail)
    with pytest.raises(OSError):worker.checkpoint_open_run(run)
    assert run.failed
    run.close_without_checkpoint()
    recovered=DurableRun(journal)
    assert recovered.session.transitions==1 and recovered.replayed==1
    recovered.stop()
