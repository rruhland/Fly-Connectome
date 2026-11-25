from types import SimpleNamespace

import numpy as np

from scripts.experimental.tt56_postures import Recovery, coverage, mpc


def test_approach_cost_checks_worst_lane_and_preserves_weighted_coverage():
    targets=np.array([[1.,0,0,0,0],[0.,0,0,0,0],[-1.,0,0,0,0.]])
    lane=np.array([0,1,2]); face=np.ones(3)
    fixed=coverage(targets[0],targets,lane,face,np.ones(5))
    balanced=coverage(targets[1],targets,lane,face,np.ones(5))
    assert balanced["score"]<fixed["score"]
    assert balanced["lanes"]==[1.,0.,1.]


def test_same_frame_cannot_advance_recovery_and_inbound_cancels(monkeypatch):
    orig=Recovery.__mro__[1]
    monkeypatch.setattr(orig,"begin",lambda self:None)
    monkeypatch.setattr(orig,"step",lambda self,t,eye,x,q,ref,base:base)
    monkeypatch.setattr(Recovery,"ready_position",lambda self,q:(0,np.full(5,.10)))
    policy=Recovery.__new__(Recovery)
    policy.kind="hub"
    policy.begin()
    y=np.zeros(6)
    x=np.array([0.,-.88,1.01,0.,-3.1,1.2])
    f=SimpleNamespace(time=0.,rms=.42,depth_sigma=.04,
                      advance=lambda now:(np.array([.05,-.45,1.14]),np.array([0.,3.7,1.4])))
    eye=SimpleNamespace(latest=f)
    assert np.array_equal(policy.step(0,eye,x,y,y,y),y)
    assert np.array_equal(policy.step(1,eye,None,y,y,y),y)
    assert np.array_equal(policy.step(2,eye,None,y,y,y),y)
    f.time=2*mpc.DT
    changed=policy.step(3,eye,None,y,y,y)
    assert np.allclose(changed[:5],.08)
    assert policy.passes[0]["moves"]==1
    assert np.array_equal(policy.step(4,eye,x,y,y,y),y)
    assert policy.goal is None
