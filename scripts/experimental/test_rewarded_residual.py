import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import pong_rewarded_residual as r
from run_rewarded_residual_pong import choose_witness


def test_centered_preference_target_makes_observed_action_maximal():
    q=np.array([[3.,2.,1.],[-2.,4.,3.],[1.,1.,1.]])
    actions=np.array([2,0,1]);d,changed=r.preference_targets(q,actions,.2)
    assert changed.all() and np.allclose(d.sum(1),0)
    assert np.array_equal(np.argmax(q+d,axis=1),actions)
    assert np.allclose((q+d)[np.arange(3),actions]-np.max(np.where(np.eye(3)[actions],-np.inf,q+d),axis=1),.2)


def test_phase_episode_weights_do_not_reward_long_trajectories():
    cases=[dict(phase='left',X=np.ones((1,2))),dict(phase='left',X=np.ones((3,2))),dict(phase='right',X=np.ones((8,2)))]
    w=r.episode_weights(cases);assert np.isclose(w[0],.25) and np.isclose(w[1:4].sum(),.25) and np.isclose(w[4:].sum(),.5)


def test_ridge_finite_centered_fits_preference_and_preserves_orthogonal_state():
    theta=np.array([[1.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    positives=[dict(phase='left',X=np.array([[1.,0.,0.]]),actions=np.array([1]))]
    preservation=[dict(phase='right',X=np.array([[0.,1.,0.]]),actions=np.array([0]))]
    buffer=np.eye(3);w=np.full(3,1/3)
    delta,info=r.fit_residual(theta,positives,preservation,buffer,w)
    assert np.isfinite(delta).all() and np.allclose(delta.sum(0),0) and np.allclose(delta[:,1:],0)
    assert info['requested_actions_before_fraction']==0 and info['requested_actions_after_fit_fraction']==1
    diag=r.action_diagnostics(theta,delta,positives,preservation)
    assert diag['requested_actions_after_cap_weighted_fraction']==1 and diag['preservation_action_change_weighted_fraction']==0


def test_positive_direction_metric_and_rescaling_bound_are_explicit():
    theta=np.array([[.001],[0.],[0.]])
    delta=np.array([[-.01],[.01],[0.]])
    X=np.ones((8,1));w=np.full(8,1/8)
    assert r.positive_disagreement(theta,delta,X,w,1.)==1 and r.positive_disagreement(theta,delta,X,w,-1.)==0
    applied,info=r.cap_residual(theta,delta,X,w,X,target=.1,maximum_rms=.10)
    assert info['applied_function_rms']<=.1000000001 and 0<info['achieved_disagreement']<=1 and info['alpha']>0
    # A nonzero fit can still fail to change greedy actions within the RMS cap.
    strong=np.array([[10.],[0.],[0.]])
    capped,small=r.cap_residual(strong,delta,X,w,X,target=.1,maximum_rms=.01)
    assert small['achieved_disagreement']==0 and small['cap_limited'] and small['applied_function_rms']<=.0100000001


def test_closest_rewarded_witness_uses_distance_then_stable_name():
    options=[dict(weights='b.npy',distance=.2),dict(weights='a.npy',distance=.2),dict(weights='c.npy',distance=.3)]
    assert choose_witness(options)['weights']=='a.npy'
