import numpy as np

from scripts.experimental.tt48_precise_receive import (
    Arc,
    DT,
    backwards,
    sphere_correct,
)
from scripts.experimental.tt18_view import CameraFit
from scripts.experimental.tt42_camera_precision import pack
from scripts.experimental.tt29_vision import Rig


def _rig():
    # Two fixed surveyed pinholes facing +Y with world-Z up.
    r = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    return Rig({name: CameraFit(np.array([x, -2., 1.2]), r, 915., 1536, 1152)
                for name,x in (("left",-.8),("right",.8))},None)


def _points(rig, xy):
    # Exact noiseless test silhouette centers, not actual simulator states.
    from scripts.experimental.tt29_rally_physics import BALL_RADIUS
    all_pix=[]
    for c in rig.fixed.values():
        projected=c.project(xy)
        all_pix.append(c.center+(projected[:,:2]-c.center)/(1.-(BALL_RADIUS/projected[:,2:])**2))
    return np.stack(all_pix,axis=1)


def test_sphere_projection_undoes_elliptic_center_offset():
    rig=_rig()
    p=np.array([[.09,-.54,.92],[-.27,-.38,.98]])
    pix=_points(rig,p)
    corrected=sphere_correct(pack(rig),pix,p)
    actual=np.stack([c.project(p)[:,:2] for c in rig.fixed.values()],axis=1)
    assert np.allclose(corrected,actual,atol=1e-10)


def test_causal_arc_recovers_noiseless_timed_pose_and_velocity_then_abstains_if_stale():
    rig=_rig()
    model=Arc(pack(rig))
    times=np.arange(-12,1)*DT
    pv=np.array([-.08,-.48,.90,.42,-4.51,-1.63])
    p=backwards(pv,np.zeros(3),times)
    for t,xy in zip(times,_points(rig,p),strict=True):
        model.add(t,xy)
    prior=pv+np.array([.0012,-.001,-.002,.013,-.016,.021])
    out=model.fit(0.,prior,spin_weight=0.,accurate=True)
    assert out is not None
    assert np.linalg.norm(out["pv"][:3]-pv[:3]) < 8e-5
    assert np.linalg.norm(out["pv"][3:]-pv[3:]) < .0011
    assert model.fit(.05,prior) is None
