import numpy as np
from pong_behavior_preservation import half_margin_bound,preserved_action_fraction,required_guard_margin
from pong_working_set_qp import solve_constrained


def test_radial_bound_uses_same_behavior_margin_as_constraints():
    theta=np.array([[1.],[0.],[-1.]]);delta=np.array([[-.5],[.5],[0.]])
    c=dict(seed=1,X=np.ones((1,1)),actions=np.array([0]),q=theta.T)
    old,_=half_margin_bound(theta,delta,[c]);new,_=half_margin_bound(theta,delta,[c],margin_cap=.1)
    assert np.isclose(old,.5) and np.isclose(new,.9) and preserved_action_fraction(delta,new,[c])==1.
    assert np.allclose(required_guard_margin([0,.01,2],.001),[0,.001,.001])


def test_looser_positive_gap_preserves_action_and_improves_same_objective():
    theta=np.array([[1.,0.],[0.,0.],[-1.,0.]]);X=np.eye(2);w=np.array([.5,.5])
    p=[dict(phase='left',X=X[:1],actions=np.array([1]))];guard=dict(seed=1,X=X[:1],actions=np.array([0]),q=X[:1]@theta.T)
    d0,r0=solve_constrained(theta,p,[],[guard],X,w)
    d1,r1=solve_constrained(theta,p,[],[guard],X,w,guard_margin_cap=.01)
    q=guard['q']+guard['X']@d1.T
    assert np.argmax(q)==0 and q[0,0]-q[0,1]>=.01-1e-8
    assert r1['constrained_objective']<r0['constrained_objective'] and np.max(abs(d1.sum(0)))<1e-12
