"""Critical collector2 configuration cannot load c0 or overwrite collection1."""
import pytest
import run_smdp_collection2 as cycle


def test_frozen_c1_and_new_destination_are_passed_without_source_edits(monkeypatch):
    original=(cycle.protocol.DEST,cycle.protocol.C0,cycle.protocol.C0_SHA)
    def capture(job,deadline,spec):
        assert cycle.protocol.DEST==cycle.DEST and cycle.protocol.C0==cycle.ACTING and cycle.protocol.C0_SHA==cycle.ACTING_SHA
        return dict(model=cycle.protocol.C0_SHA,folder=str(cycle.protocol.DEST))
    monkeypatch.setattr(cycle.protocol,'task',capture)
    got=cycle.task({},0,{})
    assert got['model']==cycle.ACTING_SHA and (cycle.protocol.DEST,cycle.protocol.C0,cycle.protocol.C0_SHA)==original


def test_configuration_is_restored_on_failed_partial_worker(monkeypatch):
    original=(cycle.protocol.DEST,cycle.protocol.C0,cycle.protocol.C0_SHA)
    def fail(*args):raise RuntimeError('simulated worker exit')
    monkeypatch.setattr(cycle.protocol,'task',fail)
    with pytest.raises(RuntimeError):cycle.task({},0,{})
    assert (cycle.protocol.DEST,cycle.protocol.C0,cycle.protocol.C0_SHA)==original
