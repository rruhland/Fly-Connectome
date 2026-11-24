import mujoco
import numpy as np

from scripts.experimental.tt19_so101 import GRIP_BODY
from scripts.experimental.tt31_ready import centered
from scripts.experimental.tt37_opponent import arena
from scripts.experimental.tt41_shallow_grip import (
    cross_section_x,
    hold_check,
    morph,
    mounted,
    weld_for,
)


def test_jaw_slice_measures_real_triangle_crossings():
    triangle=np.array([[[.003,0.,0.],[.003,2.,0.],[.003,0.,2.]],
                       [[-.003,0.,0.],[-.003,2.,0.],[-.003,0.,2.]]])
    assert np.array_equal(cross_section_x(triangle,.50,.30),[-.003,.003])
    assert not len(cross_section_x(triangle,3.,4.))


def test_auxiliary_grip_is_isolated_and_reseats_only_paddle():
    ready=(.3828,-1.62,-1.43,1.28,-.95,-.1)
    with centered(ready):
        base=arena()
        try:
            pinch=hold_check(base,'stock')
            with mounted('out-16'):
                changed=arena()
                try:
                    assert all(np.array_equal(getattr(changed.model,n),v) for n,v in morph(base.model).items())
                    for i in range(2):
                        changed.reset(render=False,seed=i)
                        weld=np.array(weld_for('out-16'));m,d=changed.model,changed.data
                        assert np.allclose(m.eq_data[m.equality('tt19_grasp').id,3:10],weld)
                        grip=m.body(GRIP_BODY).id;free=m.joint('learner_paddle_free').qposadr[0]
                        assert np.allclose(d.qpos[free:free+3],d.xpos[grip]+d.xmat[grip].reshape(3,3)@weld[:3])
                        q=np.zeros(4)
                        mujoco.mju_mulQuat(q,d.xquat[grip],weld[3:])
                        assert np.allclose(d.qpos[free+3:free+7],q)
                    new=hold_check(changed,'out-16')
                    assert new['bilateral_span_m']==pinch['bilateral_span_m']
                    assert new['least_blade_to_wrist_m']>.010
                finally:
                    changed.close()
            stock=arena()
            try:
                assert np.array_equal(stock.model.eq_data,base.model.eq_data)
            finally:
                stock.close()
        finally:
            base.close()
