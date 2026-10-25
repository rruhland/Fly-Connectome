import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from pong_behavior_preservation import half_margin_bound,preserved_action_fraction


def test_half_margin_bound_preserves_all_recorded_choices():
    theta=np.array([[1.,0.],[0.,1.],[-1.,-1.]])
    X=np.eye(2);q=X@theta.T;a=np.argmax(q,axis=1)
    delta=np.array([[-2.,0.],[2.,-1.],[0.,1.]])
    cases=[dict(seed=11,X=X,q=q,actions=a)]
    bound,info=half_margin_bound(theta,delta,cases)
    assert np.isclose(bound,.125) and info['full_guard_rows']==2 and info['full_guard_episodes']==1
    changed=q+bound*(X@delta.T)
    assert changed[0,0]-changed[0,1]>=.5*(q[0,0]-q[0,1])-1e-12
    assert preserved_action_fraction(delta,bound,cases)==1.


def test_tied_known_action_can_reject_direction_without_class_claim():
    theta=np.array([[0.],[0.],[-1.]])
    X=np.ones((1,1));q=X@theta.T;delta=np.array([[-1.],[1.],[0.]])
    cases=[dict(seed=2,X=X,q=q,actions=np.array([0]))]
    bound,info=half_margin_bound(theta,delta,cases)
    assert bound==0 and preserved_action_fraction(delta,bound,cases)==1.


def test_nonadverse_direction_has_no_additional_bound():
    theta=np.array([[1.],[0.],[-1.]])
    X=np.ones((1,1));cases=[dict(seed=3,X=X,q=X@theta.T,actions=np.array([0]))]
    bound,_=half_margin_bound(theta,np.array([[1.],[0.],[-1.]]),cases)
    assert np.isinf(bound)
