import numpy as np

from scripts.experimental.tt50_conditional_prep import bounded_forecast,deep_coordinates,hard_gate
from scripts.experimental.tt50_conditional_training import ContinuousMap


def test_hard_gate_needs_early_outgoing_and_supported_leftward_low():
    f=np.array([-.106,.988,-.12,-3.51,.333])
    x=np.array([.04,0.,1.16,-.1,2.7,.3])
    assert bounded_forecast(f)
    assert deep_coordinates(f)[1]<.97
    assert hard_gate(0.,f,x)
    assert not hard_gate(.45,f,x)
    assert not hard_gate(0.,f*np.r_[-1.,1,1,1,1],x)
    assert not hard_gate(0.,f,x*np.r_[1,1,1,1,-1,1])


def test_local_affine_refuses_unsupported_forecast_and_clamps_motion():
    rows=[dict(x=[0.,1.03,0.,-3.8,.1],q=[0.]*5,target=[8.]*5)]
    m=ContinuousMap(dict(rows=rows))
    d,s=m.query(np.array(rows[0]["x"]),np.zeros(6),np.zeros(6))
    assert s>0 and np.allclose(d,[.38,.16,.25,.27,.40])
    d,s=m.query(np.ones(5)*16,np.zeros(6),np.zeros(6))
    assert s==0 and not np.any(d)
