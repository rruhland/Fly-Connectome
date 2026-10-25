import numpy as np
from pong_local_discovery import state_indices,local_mutation
from pong_constrained_residual import solve_constrained


def test_temporal_bins_and_local_margin():
    q=np.array([[3.,0.,0.],[.01,0.,0.],[2.,0.,0.],[1.,0.,0.],[.3,0.,0.],[2.,0.,0.],[4.,0.,0.],[.2,0.,0.]])
    assert state_indices(q)==[1,3,4,7]
    theta=np.array([[2.,1.],[1.,0.],[-4.,1.]]);x=np.array([1.,2.]);z=np.eye(2)
    for b in (1,2):
        d,r=local_mutation(theta,x,b,z,kappa=.2,maximum_rms=100.)
        assert r['intended_flip_achieved'] and np.isclose(r['alternative_margin'],.2)
        assert np.allclose(d.sum(0),0)
    _,r=local_mutation(theta,x,2,z,kappa=.2,maximum_rms=1e-6)
    assert r['cap_limited'] and not r['intended_flip_achieved']


def test_matching_group_has_separate_normalized_quadratic_penalty():
    theta=np.array([[1.,0.],[0.,0.],[-1.,0.]])
    p=[dict(phase='left',X=np.array([[1.,0.]]),actions=np.array([1]))];matching=[dict(phase='right',X=np.array([[1.,1.],[1.,-1.]]))]
    d,r=solve_constrained(theta,p,[],[],np.eye(2),np.array([.5,.5]),matching_preservation=matching)
    G=np.diag([1.251,.251]);B=np.zeros((3,2));B[:,0]=(1+.05/np.sqrt(3))*(np.array([0.,1.,0.])-1/3)
    expected=np.linalg.solve(G,B.T).T
    assert np.allclose(d,expected,atol=1e-7) and r['matching_preservation_rows']==2 and r['matching_preservation_weight']==.25


def test_positive_weight_scales_both_quadratic_and_target_without_diluting_matching():
    theta=np.array([[1.,0.],[0.,0.],[-1.,0.]])
    p=[dict(phase='left',X=np.array([[1.,0.]]),actions=np.array([1]))];matching=[dict(phase='right',X=np.array([[1.,1.],[1.,-1.]]))]
    d,r=solve_constrained(theta,p,[],[],np.eye(2),np.array([.5,.5]),matching_preservation=matching,positive_weight=4.)
    G=np.diag([4.251,.251]);B=np.zeros((3,2));B[:,0]=4*(1+.05/np.sqrt(3))*(np.array([0.,1.,0.])-1/3)
    assert np.allclose(d,np.linalg.solve(G,B.T).T,atol=1e-7) and r['positive_weight']==4.
