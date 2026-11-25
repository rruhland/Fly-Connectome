import numpy as np

from scripts.experimental.tt51_low_contact_practice import ContactMemory


def test_true_outcome_posterior_prefers_rewarded_drive_and_keeps_misses():
    memory=ContactMemory(np.zeros(6),learn=True)
    for arm,landed in [(12,False)]*3+[(13,True)]*3:
        memory.pending=dict(family=arm,context=[0.],plan=None)
        memory.record(landed)
    assert len(memory.rows)==6
    assert sum(r['landed'] for r in memory.rows)==3
    memory.learn=False
    before=memory.digest()
    assert memory.choose(np.zeros(16),{4})==13
    memory.record(False)
    assert memory.digest()==before


def test_geometry_mask_prevents_recalling_an_unavailable_family():
    memory=ContactMemory(np.zeros(6),learn=False,packet=dict(rows=[
        dict(family=13,landed=True,context=[0.],plan=None) for _ in range(8)]))
    assert memory.choose(np.zeros(16),{3}) in (9,10,11)
    memory.pending=None
    assert memory.choose(np.zeros(16),set()) is None
    assert memory.pending is None


def test_no_learning_or_credit_without_an_actual_selected_attempt():
    memory=ContactMemory(np.zeros(6),learn=True)
    assert memory.keep([],None)==0
    assert memory.rows==[]
    memory.pending=dict(family=13,context=[0.],plan=None)
    assert memory.keep([],None)==1
    assert len(memory.rows)==1 and memory.rows[0]['landed']
