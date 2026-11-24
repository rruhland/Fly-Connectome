import numpy as np

from scripts.experimental.tt42_inbound_prep import Feedback, early, split_passages


def bank():
    x=np.array([.02,.69,1.06,.01,-3.81,.09,.05,-1.5,-1.4,1.1,-.82])
    templates=np.tile(x,(8,1))
    templates[:,0]+=np.linspace(-.04,.04,len(templates))
    left=np.array([.44,0.,.09,-.38,-.28])
    right=-left
    actions=np.array([left,right]*4)
    r=np.array([1,0]*4)
    return x, left, (templates,actions,r,np.arange(len(r)),None,None)


def test_local_vote_chooses_a_successful_mode_not_the_average():
    x,left,experience=bank()
    expert=Feedback(experience)
    ref,note=expert.decide(x,-left)
    np.testing.assert_allclose(ref,left)
    assert note["estimated_gain"]>.05
    assert note["independent_local"]==8


def test_learning_only_stores_motion_after_a_physical_outcome():
    x,_,experience=bank()
    learner=Feedback(experience)
    trace=[]
    for k in range(40):
        p=x[:6].copy()
        p[1]=1.46-.067*k
        angles=x[6:11]
        sent=np.r_[np.ones(5)*.006*k,-.1]
        trace.append((k,p,angles,sent-.006,sent))
    assert not learner.past
    learner.heard(trace,True)
    assert learner.updates==[0,1]
    assert len(learner.past)>4
    np.testing.assert_allclose(learner.past[0][1],.036)
    learner.heard(trace,False)
    assert learner.updates==[1,1]
    assert learner.past[-1][2]==0
    frozen=Feedback(experience,online=False)
    frozen.heard(trace,True)
    assert not frozen.past


def test_preparation_scope_and_release_group_breaks():
    x,_,_=bank()
    assert early(x[:6])
    a=x[:6].copy();a[1]=-.11
    assert not early(a)
    a=x[:6].copy();a[4]=3.6
    assert not early(a)
    seq=np.tile(x,(4,1))
    seq[:,1]=[.4,.34,1.02,.97]
    np.testing.assert_array_equal(split_passages(seq),[1,1,2,2])
