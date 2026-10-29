from collections import deque
import hashlib
import numpy as np
import pytest
from compact_public_pong import fitness_key,evaluate,time_bins,CompactEvaluator,new_controller
from pong_relational_feedback import RelationalPolicy
from pong_relational_persistent_binding import PersistentRelationalPolicy


def model_file(tmp_path):
    path=tmp_path/'policy.npz'
    PersistentRelationalPolicy(RelationalPolicy(3,0,1,0,1,np.zeros(5),np.ones(4)),False).save(path)
    return path


def test_duration_and_model_are_part_of_exact_fitness_identity(tmp_path):
    path=model_file(tmp_path)
    assert len({fitness_key(path,1,2400,'runtime'),fitness_key(path,1,7200,'runtime'),
        fitness_key(None,1,2400,'runtime'),fitness_key(path,2,2400,'runtime'),
        fitness_key(path,1,2400,'other-runtime')})==5


def test_time_strata_assign_boundary_events_to_elapsed_window():
    events=[dict(tick=t,hit=h,first=i==0) for i,(t,h) in enumerate(
        [(2400,True),(2401,True),(4800,False),(4801,True),(7200,True)])]
    bins=time_bins(events,60.)
    assert [b['own_opportunities'] for b in bins]==[1,2,2]
    assert [b['later_returns'] for b in bins]==[0,1,2]
    assert [b['exposure_seconds'] for b in bins]==[20,20,20]
    partial=time_bins(events[:2],25.)
    assert [b['exposure_seconds'] for b in partial]==[20,5,0]


def test_expired_job_has_no_native_initial_observation(tmp_path,monkeypatch):
    import compact_public_pong as worker
    def forbidden(*args,**kwargs):raise AssertionError('expired job constructed a native evaluator')
    monkeypatch.setattr(worker,'CompactEvaluator',forbidden)
    r=evaluate(model_file(tmp_path),1,7200,tmp_path,0.,3000,'runtime')
    assert r['not_started'] and not r['fitness_eligible'] and r['transitions']==r['initial_cameras']==0
    with pytest.raises(ValueError):evaluate(None,1,2400,tmp_path,0.,1001,'runtime')


def test_streaming_controller_check_uses_separate_models_and_detects_mismatch(tmp_path,monkeypatch):
    import compact_public_pong as worker
    path=model_file(tmp_path);primary=PersistentRelationalPolicy.load(path);shadow=PersistentRelationalPolicy.load(path)
    assert not np.shares_memory(primary.original.coefficients,shadow.original.coefficients)
    c=object.__new__(CompactEvaluator);c.model=primary;c.shadow_model=shadow
    c.controller=new_controller(primary);c.shadow=new_controller(shadow)
    c.digest=hashlib.sha256();c.ring=deque(maxlen=64);c.pending_entry=None;c.mismatch=None;c.transitions=0;c.scaler=None
    c.latest_encoded=dict(entities=[dict(id=i,normalized_position=[y,x],observed_velocity=[0.,0.],
        observed=True,velocity_known=True,observation_age_seconds=0.,velocity_age_seconds=0.,association_strength=1.)
        for i,y,x in ((1,.1,.1),(2,.9,.5),(3,.2,.9))])
    class Baseline:
        def choose(self,*args):return 0,np.array([1.,0.,0.])
    c.rbf_readout=Baseline();monkeypatch.setattr(worker,'policy_features',lambda *args:(np.array([0]),np.array([1.])))
    action,_=c._choose_policy();assert action==2 and c.ring[-1]['state_before']['bound_ids'] is None
    c.controller.observe(0);c.shadow.observe(0);c.transitions=1
    old=c.shadow.choose
    def wrong(*args):
        _,decision=old(*args);return 0,decision
    c.shadow.choose=wrong
    with pytest.raises(AssertionError):c._choose_policy()
    assert c.mismatch is not None
