import numpy as np

from scripts.experimental.tt55_midline_flat import fit
from scripts.experimental.tt55_midline_sweep import velocities


class FixedAxis:
    low=np.full(5,-1.)
    high=np.full(5,1.)
    normal=np.array([0.,.8,.6])

    def pose(self,q,*,jacobian=False):
        if not jacobian:
            return q[:3],self.normal
        return q[:3],self.normal,np.c_[np.eye(3),np.zeros((3,2))],np.zeros((3,5))


def test_contact_geometry_trades_center_offset_for_a_fixed_face():
    ik=FixedAxis()
    p=np.array([.22,-.35,.51]); q0=np.zeros(5)
    _,gap,rad,n=fit(ik,p,ik.normal,1,q0,.045)
    assert abs(gap-.026)<1e-5 and rad<.046
    np.testing.assert_allclose(n,ik.normal)


def test_tangential_sweep_obeys_signed_joint_constraint_at_limit():
    ik=FixedAxis()
    motion,n,act=velocities(ik,np.array([0.,1.,0.,0.,0.]),1,.80,1.)
    assert -4.<=motion[1]<=.0101
    np.testing.assert_allclose(n,ik.normal)
    assert abs(act[1]-.010) < .002
