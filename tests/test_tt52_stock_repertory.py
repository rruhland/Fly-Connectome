from types import SimpleNamespace

import numpy as np

from scripts.experimental.tt52_stock_repertory import Joined, fresh_bounce, mpc, supported_choice


def test_supported_stock_or_recent_optional_stroke_cannot_be_displaced():
    a=SimpleNamespace(best=np.zeros((30,6)),tick=24,lease=7,last_extra=-999,
                      notes=[dict(tick=24,chosen='credited',robust=1)])
    assert supported_choice(a,28)
    assert not supported_choice(a,32)
    a.notes=[]
    assert not supported_choice(a,28)
    a.last_extra=24
    assert supported_choice(a,28)


def test_public_barrier_requires_a_fresh_postbounce_measurement():
    assert fresh_bounce([(24*mpc.DT,)],25,True)
    assert not fresh_bounce([(21*mpc.DT,)],25,True)
    assert not fresh_bounce([(24*mpc.DT,)],25,False)


def test_optical_low_template_requires_underlying_encoder_support():
    class Pattern:
        def proposals(self,p,path,x):
            assert np.allclose(p,[-.1,.989,-.13,-3.8,.31])
            return [('example',np.zeros((mpc.HORIZON,6)))]
    field=Joined(None,Pattern())
    t=np.array([[.13,-.1,-1.1,.989,-.13,-3.8,.31,0,0,0]])
    x=np.array([-.08,-.55,.832,-.13,-3.8,1.8]);z=np.zeros(5)
    assert [name for name,_ in field.propose(x,t,(z,z,z))]==['low/example']
    assert not field.propose(x,t,(z+1.0,z,z))
    t[0,3]=1.06
    assert not field.propose(x,t,(z,z,z))
