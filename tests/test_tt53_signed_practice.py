import numpy as np

from scripts.experimental.tt53_signed_practice import N, issue, select, source_bank, surrogate


def sensed(v):
    return {"sensed":None if v is None else {"value":v}}


def test_signed_reply_favors_empirically_better_direction():
    records=dict(zip(("axis0+","axis0-","axis1+","axis1-"),map(sensed,(0.3,0.0,-0.2,0.8))))
    d,note=select(records)
    assert np.array_equal(d,[0,-1])
    assert note["signed"]
    assert note["finite_difference"]<0
    records["axis1-"]=sensed(None)
    d,note=select(records)
    assert np.array_equal(d,[1,0])
    assert note["signed"]
    records["axis0-"]=sensed(None)
    assert not select(records)[1]["signed"]


def test_correction_is_bounded_then_keeps_original_command_family():
    failure=np.zeros((N+5,6))
    failure[N:,0]=.20
    basis=np.ones((2,N,5))*5
    controls=issue(failure,basis,np.array([1.2,-.3]),np.ones(6)*.1,
                   np.full(6,-2.),np.full(6,2.),np.zeros(6))
    assert np.all(abs(controls)<=.8)
    assert np.array_equal(controls[:,5],failure[:,5])
    assert np.array_equal(controls[N:],failure[N:])


def test_holdout_excludes_every_matched_fork_of_that_release():
    r={"observed":[.1,-.7,1.,0.,-3.,1.],"q":[0.]*6,"ref":[0.]*6,
       "loss":np.zeros((N,6)).tolist(),"win":np.ones((N,6)).tolist()}
    data=[dict(r,id=k) for k in (900,3,900,7)]
    assert [x["id"] for x in source_bank(data,900)]==[3,7]


def test_optical_response_favors_forward_supported_arc_over_backward_touch():
    p=np.array([0.,-1.52,1.02])
    score_up,_=surrogate(p,np.array([.1,4.1,3.2]))
    score_back,_=surrogate(p,np.array([-.2,-.8,-3.0]))
    assert score_up>score_back
