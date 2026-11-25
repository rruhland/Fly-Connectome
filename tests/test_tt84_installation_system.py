import numpy as np
from scipy.spatial.transform import Rotation

from scripts.experimental import tt84_installation_play as play
from scripts.experimental import tt84_installation_system as system


def test_joint_and_world_blade_transform_matches_compiled_supported_fixture():
    v=[.11972,.00561,17.96411,-7.59597,-4.00054]
    q=np.array([-.02,-1.70,-1.62,1.53,-.72])
    a=system.MP.arena(width=64,height=48,detail='cad')
    try:
        origin,foot=system.stock_foot(a)
        m=system.layout(v,origin,foot)
        assert m is not None
        p,R=system.pose.FullPose(a).frame(q)
        arrays=system.invariants(a.model)
        with system.installed(v,m):
            b=system.MP.arena(width=64,height=48,detail='cad')
            try:
                bp,bR=system.pose.FullPose(b).frame(q)
                np.testing.assert_allclose(bp,np.asarray(m['rotation'])@(p-origin)+m['base'],atol=1e-8)
                np.testing.assert_allclose(bR,np.array(m['rotation'])@R,atol=1e-8)
                assert system.physical_support(b)['safe']
                assert all(np.array_equal(v,system.invariants(b.model)[k]) for k,v in arrays.items())
            finally:b.close()
    finally:a.close()


def test_sensor_registration_keeps_displacements_translation_free():
    R=Rotation.from_euler('y',22,degrees=True).as_matrix()
    old=np.array([0.,-1.4,.77]); new=np.array([.13,-1.39,.80])
    x=np.array([.03,-1.09,1.02,.3,-3.5,.6]); t=np.r_[R@(x[:3]-old)+new,R@x[3:]]
    reg=play.Registration(old,new,R)
    np.testing.assert_allclose(reg.local(t),x)
    delta=np.array([.03,.04,.03,.08,-.04,.01]); measured=np.r_[R@delta[:3],R@delta[3:]]
    f=np.asarray([[*t,*np.arange(5)],[*measured,*np.arange(5)]]).ravel()
    out=reg.feature(f,np.ones(11)).reshape(2,11)
    np.testing.assert_allclose(out[1,:6],delta)
