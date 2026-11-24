import numpy as np

from scripts.experimental.tt37_rebound import accel
from scripts.experimental.tt41_early_optics import CausalBounce


def law(tmp):
    incoming=np.tile([.20,-3.60,-2.30],(20,1))
    delta=np.tile([-.05,.164,-.21],(20,1))
    f=tmp/"rebound.npz"
    np.savez(f,rebound_states=incoming,rebound_delta=delta,rebound_z=np.full(20,.774))
    return CausalBounce(f)


def ball(t, bounced=True):
    center=np.array([-.08,-.74,.774])
    v=np.array([.20,-3.60,-2.30] if t<0 or not bounced else [.15,-3.436,2.09])
    return center+v*t+.5*accel(v)*t*t


def test_bounce_appears_only_when_there_is_current_visual_evidence(tmp_path):
    obj=law(tmp_path)
    for t in np.arange(-.188,.004,1/30):
        assert obj.update(t,ball(t)) is None
        assert not obj.active
    endpoint=.012
    result=obj.update(endpoint,ball(endpoint))
    assert obj.active and obj.impact.post_frames==1
    assert obj.impact.time < endpoint
    assert np.linalg.norm(result[:3]-ball(endpoint))<.004
    assert result[5] > 1.35


def test_downward_free_arc_does_not_confirm_a_bounce(tmp_path):
    obj=law(tmp_path)
    for t in np.arange(-.188,.066,1/30):
        assert obj.update(t,ball(t,bounced=False)) is None
    assert not obj.active
