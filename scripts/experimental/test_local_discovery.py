import numpy as np
from pong_local_discovery import state_indices,local_mutation,drop_guard_conflicts
from pong_constrained_residual import solve_constrained


def test_temporal_bins_and_local_margin():
    q=np.array([[3.,0.,0.],[.01,0.,0.],[2.,0.,0.],[1.,0.,0.],[.3,0.,0.],[2.,0.,0.],[4.,0.,0.],[.2,0.,0.]])
    assert state_indices(q)==[1,3,4,7]
    assert state_indices(q,midpoints=True)==[1,3,4,5,7]
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


def test_guard_conflicts_are_exact_only_and_do_not_discard_the_episode():
    guards=[dict(X=np.array([[1.,0.]]),actions=np.array([0]))]
    cases=[dict(seed=9,phase='left',X=np.array([[1.,0.],[1.,1e-12],[2.,0.]]),actions=np.array([1,1,2]))]
    kept,report=drop_guard_conflicts(cases,guards)
    assert report['dropped_rows']==1 and report['dropped_episodes']==0
    assert np.array_equal(kept[0]['actions'],[1,2]) and kept[0]['X'][0,1]==1e-12


def test_rewarded_witness_choice_uses_current_relative_distance(tmp_path):
    import json
    from run_local_repair_pong import choose_donor_entries,base
    (tmp_path/'episodes').mkdir();entries=[]
    for name,scale in [('far',2.),('near',.2)]:
        np.save(tmp_path/(name+'.npy'),np.full((3,2),scale));receipt=tmp_path/'episodes'/(name+'.json')
        receipt.write_text(json.dumps(dict(weights_file=name+'.npy',weights_sha256=base.sha(tmp_path/(name+'.npy')),report=dict(hits=1))))
        entries.append(dict(seed=1,phase='left',feature_file=name+'.npz',reward_receipt=str(receipt)))
    chosen=choose_donor_entries(tmp_path,entries,np.eye(2),np.zeros((3,2)))
    assert len(chosen)==1 and chosen[0]['feature_file']=='near.npz' and np.isclose(chosen[0]['current_function_distance'],.2)
