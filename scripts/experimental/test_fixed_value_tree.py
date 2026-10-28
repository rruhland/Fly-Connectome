"""Meaningful support, terminal-backup and topology invariants for tree pilot."""
import time
import numpy as np
from pong_fixed_value_tree import build_forest, leaves_batch, fit_values, supported_values
from run_rally_fixed_value_tree import allocate


def test_fixed_topology_and_routes():
    F=np.random.default_rng(13).normal(size=(4096,711))
    A,n=build_forest(F,np.random.default_rng(2026102882))
    B,m=build_forest(F,np.random.default_rng(2026102882))
    assert n==m and np.array_equal(A,B)
    L=leaves_batch(F,A)
    for t in range(32):
        counts=np.bincount(L[:,t],minlength=n)
        assert (counts[counts>0]>=16).all()
    assert (L>=0).all() and (L<n).all()


def test_exact_weighted_backup_support_terminal_cutoff():
    # Each tree has one leaf, four equal-weight executed rows per action.
    current=np.tile(np.arange(32),(12,1));successor=current.copy()
    actions=np.repeat(np.arange(3),4);rewards=np.r_[np.ones(4),np.zeros(8)]
    discounts=np.zeros(12);weights=np.full(12,1/12)
    V,count,weight,ess,eligible,report=fit_values(current,successor,actions,rewards,discounts,weights,32,time.monotonic()+30)
    np.testing.assert_allclose(V,np.tile([1,0,0],(32,1)))
    np.testing.assert_allclose(ess,4.)
    assert eligible.all() and report['completed_sweeps']==256
    q,support,_=supported_values(successor,V,eligible)
    assert support.all() and np.argmax(q,axis=1).tolist()==[0]*12
    # A nonterminal successor with no data is an explicit coverage cutoff.
    unseen=np.tile(np.arange(32,64),(12,1))
    V,_,_,_,_,report=fit_values(current,unseen,actions,rewards,np.ones(12)*.99,weights,64,time.monotonic()+30)
    np.testing.assert_allclose(V[:32],np.tile([1,0,0],(32,1)))
    assert report['coverage_cutoff_backup_rows']==12 and report['natural_terminal_rows']==0


def test_quota_keeps_protected_and_exhausts_exact_budget():
    out=allocate([50,10,100],[2,1,25],70)
    assert out.sum()==70 and (out>=[2,1,25]).all() and (out<=[50,10,100]).all()
