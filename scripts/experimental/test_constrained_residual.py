import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from pong_constrained_residual import solve_constrained,CONTRAST
from pong_rewarded_residual import fit_residual


def problem():
    theta=np.array([[1.,1.],[0.,0.],[-1.,-1.]])
    positive=[dict(phase='left',X=np.array([[1.,0.]]),actions=np.array([1]))]
    preservation=[dict(phase='right',X=np.array([[0.,1.]]),actions=np.array([0]))]
    X=np.eye(2);w=np.full(2,.5)
    return theta,positive,preservation,X,w


def test_unconstrained_contrast_basis_matches_exact_ridge():
    theta,p,s,X,w=problem();a,info=solve_constrained(theta,p,s,[],X,w);b,_=fit_residual(theta,p,s,X,w)
    assert np.allclose(CONTRAST.T@CONTRAST,np.eye(2)) and np.allclose(a,b,atol=1e-7)
    assert np.allclose(a.sum(0),0,atol=1e-12) and info['variables']==4


def test_soft_conflicting_target_remains_feasible_and_preserves_guard():
    theta,p,s,X,w=problem();guard=dict(seed=1,X=p[0]['X'],actions=np.array([0]),q=p[0]['X']@theta.T)
    delta,info=solve_constrained(theta,p,s,[guard],X,w)
    q=guard['q']+guard['X']@delta.T
    assert info['maximum_half_margin_violation']<1e-7 and info['constrained_objective']<=info['zero_residual_objective']+1e-7
    assert q[0,0]-q[0,1]>=.5-1e-7 and np.argmax(q)==0 and info['guard_action_fraction']==1.


def test_distinct_repair_can_change_action_while_preserving_success():
    theta,p,s,X,w=problem();guard=dict(seed=2,X=s[0]['X'],actions=np.array([0]),q=s[0]['X']@theta.T)
    delta,info=solve_constrained(theta,p,s,[guard],X,w)
    assert np.argmax((theta+delta)@p[0]['X'][0])==1
    assert np.argmax(guard['q']+guard['X']@delta.T)==0 and info['requested_actions_after_fit_weighted_fraction']==1.
