import numpy as np

from scripts.experimental.tt41_deadline import RATE, SLEW, TurnGate, bound
from scripts.experimental.tt41_deadline_warm import Outbound, WARM_AXES, warm_ref


def feed(gate, ys):
    return [gate.update(2*k, None if y is None else [0.,y,1.],1./60) for k,y in enumerate(ys)]


def test_rgb_return_gate_requires_earlier_outbound_and_consecutive_centers():
    assert not any(feed(TurnGate(),[.50,.44,.25,.08]))
    assert feed(TurnGate(),[-.20,-.13,-.04,.01,.08,.15,.22,.11,-.02])[-2:]==[False,True]
    assert feed(TurnGate(),[.2,.3,.4,.3,None,.1,0.])[-2:]==[False,False]


def test_outbound_ready_starts_only_after_visible_flight_moves_away():
    assert feed(Outbound(),[-1.1,-.85,-.66,-.51]) == [False,False,False,True]
    assert not any(feed(Outbound(),[-.4,-.2,None,.1,.3]))


def test_bounded_motor_targets_respect_rate_and_slew_at_both_rails():
    q0=np.array([.1,-.3])
    last=np.zeros(2)
    stride=np.array([.082,.091])
    lo=-np.ones(2)
    hi=np.ones(2)
    goals=np.r_[np.tile([1.7,-1.7],(38,1)),np.tile([-1.7,1.7],(38,1))][None]
    q=bound(goals,q0,last,stride,lo,hi)
    rates=np.diff(np.concatenate([q0[None,None,:],q],axis=1),axis=1)[0]/stride
    assert np.all(q>=lo) and np.all(q<=hi)
    assert np.max(abs(rates))<=RATE+1e-10
    assert np.max(abs(np.diff(np.vstack([last,rates]),axis=0)))<=SLEW+1e-10
    np.testing.assert_allclose(q[0,-1],[-1.,1.])


def test_common_ready_preserves_stock_lateral_pan_tilt_and_grip():
    def proof(angles):
        return {'wins':[{'chosen':{'variant':{'approach':angles}}}]}
    docs={'long-low-flipped':{'rows':[proof([4.,1.,1.,4.,2.])]},
          'long-low-alternate-strict':{'rows':[proof([-4.,1.,1.,-4.,2.]),proof([0.,1.,1.,0.,2.])]}}
    changed=warm_ref(np.zeros(6),docs)
    assert tuple(np.flatnonzero(changed))==WARM_AXES
    assert np.all(changed[list(WARM_AXES)]>0.)
