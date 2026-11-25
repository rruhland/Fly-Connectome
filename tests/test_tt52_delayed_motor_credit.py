from collections import Counter

import numpy as np

from scripts.experimental.tt52_delayed_motor_credit import Credit, grades, sham_values


def test_eligibility_advances_only_on_an_observed_future_native_score():
    c=Credit()
    c.begin(42)
    c.reward(1)
    assert c.values()=={}
    c.reward(2)
    assert c.values()=={(42,1):1.}
    c.reward(3)
    assert c.grade(42)=={(42,1):(1,1),(42,2):(1,0),(42,3):(0,0)}
    c.loss()
    assert c.grade(42)=={(42,1):(1,1),(42,2):(1,-1),(42,3):(-1,0)}
    assert c.values()[(42,1)] > c.values()[(42,2)] > c.values()[(42,3)]
    c.begin(43)
    c.reward(1)
    assert (43,1) not in c.values()
    assert c.values()[(42,3)]==-1.


def test_sham_keeps_outcomes_and_censoring_within_each_game():
    x={(42,1):1.,(42,2):.3,(42,3):-1.,(46,1):1.,(46,2):1.,(46,3):-1.}
    y=sham_values(x)
    assert y==sham_values(x)
    assert x.keys()==y.keys()
    for game in (42,46):
        assert Counter(v for (g,_),v in x.items() if g==game)==Counter(v for (g,_),v in y.items() if g==game)


def test_repeated_success_needs_independent_physical_strokes_not_one_game():
    t=np.zeros((6,16))
    a=np.r_[np.zeros((3,15)),np.full((3,15),1.)]
    y=np.array([1.,1.,1.,-1.,-1.,-1.])
    d=np.zeros(6)
    valid=np.ones(6,bool)
    w,supported=grades(t,a,y,np.arange(6),d,valid)
    assert np.all(supported)
    assert np.all(w[:3]>0) and np.all(w[3:]<0)

    w,supported=grades(t,a,y,np.zeros(6,dtype=int),d,valid)
    assert not np.any(supported) and not np.any(w)

    valid[0]=False
    w,supported=grades(t,a,y,np.arange(6),d,valid)
    assert not supported[0] and w[0]==0
