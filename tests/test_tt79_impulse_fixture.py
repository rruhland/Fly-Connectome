from types import SimpleNamespace

import numpy as np

from scripts.experimental import tt79_impulse_fixture as study


def test_patch_jacobian_uses_rotation_about_actual_contact_point():
    J=np.zeros((3,5));W=np.zeros((3,5));W[1,0]=1.
    ik=SimpleNamespace(low=np.full(5,-1.),high=np.full(5,1.),
                       frame=lambda q,j:(np.zeros(3),np.eye(3),J,W))
    _,v,err,*_=study.twist_patch(ik,np.zeros(5),np.array([.05,0,0]),np.array([0.,0.,1.]),
                                np.array([0.,0.,.15]),np.array([-3.,0,0]))
    np.testing.assert_allclose(v,[0.,0.,.15],atol=1e-6)
    assert err<1e-6


def test_flat_riser_leaves_original_table_and_paddle_mass_and_moves_only_base():
    remembered=None
    for option in ('stock','riser18','stock'):
        with study.mounted(option):
            a=study.MP.arena(width=64,height=48,detail='cad')
            try:
                fixed=study.fixed_arrays(a.model); p=a.model.body('learner_base').pos.copy()
                assert study.support(a,option)['safe']
                if remembered is None:
                    remembered=p,fixed
                else:
                    assert all(np.array_equal(v,fixed[k]) for k,v in remembered[1].items())
                    np.testing.assert_allclose(p-remembered[0],[0.,0.,.018 if option=='riser18' else 0.],atol=1e-9)
            finally:a.close()
