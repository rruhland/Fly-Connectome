import numpy as np

from scripts.experimental.tt31_ready import centered
from scripts.experimental.tt37_opponent import arena
from scripts.experimental.tt41_shallow_clamp import ADAPTER, pitched, static_check, unchanged


def test_same_clamp_wedge_seats_original_arm_and_keeps_original_dynamics():
    ready=(.3828,-1.62,-1.43,1.28,-.95,-.1)
    with centered(ready):
        base=arena()
        try:
            with pitched(6.):
                alternate=arena()
                try:
                    stock=unchanged(base.model,0.)
                    changed=unchanged(alternate.model,6.)
                    assert all(np.array_equal(changed[k],v) for k,v in stock.items())
                    assert static_check(alternate,6.)['safe']
                    assert np.allclose(base.model.body('learner_base').pos[:2],
                                       alternate.model.body('learner_base').pos[:2])
                    assert not np.array_equal(base.model.body('learner_base').quat,
                                              alternate.model.body('learner_base').quat)
                    assert alternate.model.geom(ADAPTER).id>=0
                finally:
                    alternate.close()
        finally:
            base.close()
