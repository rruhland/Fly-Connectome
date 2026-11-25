import numpy as np

from scripts.experimental.tt44_continuous_credit import sensed_look
from scripts.experimental.tt44_continuous_credit_fit import Local, first_public


def test_bounded_sensed_look_preserves_input_and_limits_preview():
    x=np.array([.04,.5,.95,.3,-4.,.4])
    y=sensed_look(x,200)
    assert y[1]>=-.56
    assert y[1]<x[1]
    assert np.isclose(x[1],.5)


def test_first_public_context_precedes_issued_controls():
    f=np.zeros(31)
    f[0]=4
    f[2]=.45
    f[5]=-4.1
    f[19:]=99
    tick,x=first_public([f])
    assert tick==4 and x.shape==(16,)
    assert np.max(abs(x))<8.


def test_local_policy_needs_a_real_geographically_local_improvement():
    z=np.zeros(16)
    p=[.3,.2,-.3,0.,.1,.1]
    cases=[dict(context=z,stock=0,trials=[dict(reward=1,u=p),dict(reward=0,u=[0.]*6)])]
    local=Local(cases,[])
    assert np.allclose(local.get(z),p)
    assert not np.any(local.get(z+2))
    assert not np.any(Local([dict(context=z,stock=1,trials=[dict(reward=1,u=p)])],[]).get(z))
