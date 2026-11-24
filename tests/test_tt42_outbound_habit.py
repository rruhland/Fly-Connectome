from types import SimpleNamespace

import numpy as np

from scripts.experimental import tt42_outbound_habit as prep


def optics(tick, y, vy):
    return SimpleNamespace(
        position=np.array([0.,y,1.25]), velocity=np.array([0.15,vy,0.3]),
        time=tick/60, rms=.24, depth_sigma=.025, frames=4, bounced=False,
    )


def motor_memory(outbound):
    f=prep.feature(outbound)
    assert f is not None
    goal=np.tile([.123,.14,.17,.083,-.11,0.],(prep.SAMPLE_TICKS,1))
    return dict(features=[f,f],groups=[1,2],relative_curves=[goal,goal])


def test_needs_visible_prior_inbound():
    out=optics(16,-1.06,4.10)
    h=prep.Habit(motor_memory(out),True)
    watched=np.zeros(6)
    h.inspect(16,out,1./60,watched)
    assert not h.records
    h.inspect(18,optics(18,-1.21,-3.),1./60,watched)
    out.time=20./60
    h.inspect(20,out,1./60,watched)
    assert len(h.records)==1
    assert h.records[0]['tick']==20
    assert h.records[0]['max_offset'][0]==h.records[0]['max_offset'][4]==0.


def test_movement_is_bounded_then_releases_on_observed_inward_motion():
    out=optics(18,-1.06,4.10)
    h=prep.Habit(motor_memory(out),True)
    z=np.zeros(6)
    one=np.ones(6)
    stride=.09*one
    h.inspect(16,optics(16,-1.17,-3.),1./60,z)
    u=h.command(18,out,1./60,z,z,z,stride,-one,one)
    assert np.max(abs(u)) <= prep.RATE
    assert np.max(abs(u)) > 0.
    assert u[0]==u[4]==u[5]==0.
    h.command(32,optics(32,.43,-3.),1./60,z,z,z,stride,-one,one)
    assert h.stance is None
    assert h.records[0]['release']=='inbound_rgb'
