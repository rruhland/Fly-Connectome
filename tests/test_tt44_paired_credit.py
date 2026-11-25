import numpy as np

from scripts.experimental.tt44_paired_credit import TEACHERS, features
from scripts.experimental.tt44_paired_credit_fit import SCALE, Selector, movement


def test_paired_selector_requires_repeated_compared_gain():
    x=np.zeros((4,len(SCALE)))
    assert Selector(x,np.zeros((4,2))).choose(x[0])=="stock"
    assert Selector(x,[[1,0],[0,0],[0,0],[0,0]]).choose(x[0])=="stock"
    assert Selector(x,[[1,-1],[1,0],[0,0],[0,0]]).choose(x[0])==TEACHERS[1]
    assert Selector(x,[[1,1],[0,0],[0,0],[0,0]]).choose(x[0])==TEACHERS[1]
    assert Selector(x,[[1,1],[0,0],[0,0],[0,0]]).choose(np.ones(len(SCALE))*3.)=="stock"


def test_context_uses_only_current_images_and_encoder_references():
    x=np.arange(6,dtype=float)
    q=np.arange(6,dtype=float)
    assert features(x,q,q,x).shape==SCALE.shape
    assert np.all(features(x,q,q,x)[6:12]==0)
    assert np.all(features(x,q,q,x)[-5:]==0)


def test_movement_indexes_the_public_joint_prefix():
    f=np.zeros(31)
    f[0]=123
    g=f.copy()
    g[7]=.27
    row={"tick":123,"variants":{"stock":{"sensory":[f],"controls":[[0]]},
                               "temporal":{"sensory":[g],"controls":[[1]]}}}
    d=movement(row,"temporal")
    assert d["first_different_control"]==123
    assert np.isclose(d["maximum_early_encoder_shift_rad"],.27)
