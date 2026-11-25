import numpy as np
from scipy.spatial.transform import Rotation

from scripts.experimental import tt104_reanchor_impulse as retarget
from scripts.experimental.tt4_transfer_trial import _read


def test_relative_impulse_changes_rotate_the_face_and_leave_rigid_frame_valid():
    R=Rotation.from_euler('x',-90.,degrees=True).as_matrix()
    n=R[:,2]
    center=np.array([.16,-1.56,1.08])
    hit=np.array([.147,-1.539,1.067])
    incoming=np.array([.21,-2.9,.1])
    patch=dict(p=center,R=R,x=hit,incoming=incoming,out=np.array([-.1,3.6,1.75]),twist=np.array([.4,.2,1.,.29,.51,.22]))
    anchor=object.__new__(retarget.Anchor)
    same=anchor.relabel(patch,hit,incoming,1.,1.)
    np.testing.assert_allclose(same[0],center,atol=1e-10)
    np.testing.assert_allclose(same[1],R,atol=1e-10)
    np.testing.assert_allclose(same[2],patch['twist'],atol=1e-10)
    off=anchor.relabel(patch,hit+.034,incoming+np.array([.40,-.21,-.05]),.80,.80)
    np.testing.assert_allclose(off[1]@off[1].T,np.eye(3),atol=1e-10)
    assert abs(np.linalg.det(off[1])-1.)<1e-10
    assert .01<np.linalg.norm(off[1][:,2]-n)<.21
    assert .01<np.linalg.norm(off[2][3:]-patch['twist'][3:])<.20


def test_new_control_includes_an_original_successful_low_and_disjoint_routine_cases():
    p=_read(retarget.PLAN)
    ids=[x['id'] for x in p['held']]
    assert len(set(ids))==len(ids)
    assert not set(ids)&set(p['excluded'])
    assert any(c['role']=='low' and c['scored'] for c in p['held'])
    assert sum(c['role']=='miss' for c in p['held'])==5
