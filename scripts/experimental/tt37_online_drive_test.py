"""Checks for signed optical credit and bounded continuously selected stroke controls."""
import numpy as np

from scripts.experimental.tt37_online_drive import LocalDrive, grade


def test_signed_reversal_can_be_graded_even_without_positive_forward_flight():
    near = grade(False, {"outgoing": [0., -1.5, .5]}, -.30)
    far = grade(False, {"outgoing": [0., 2.0, .5]}, .24)
    assert 0 < near < far < grade(True, None, None)
    assert grade(False, None, None) is None


def test_continuous_kernel_does_not_change_unsupported_room_position():
    rng=np.random.default_rng(21)
    memory=LocalDrive()
    x=np.zeros(10)
    trial=[.37,-.43]
    assert np.all(memory.choose(x,rng,False)[0] == 0.)
    memory.observe([dict(context=x.tolist(),parameter=trial,value=1.0)])
    np.testing.assert_allclose(memory.choose(x,rng,False)[0],trial)
    unseen=x.copy()
    unseen[0] += 1.5
    assert np.all(memory.choose(unseen,rng,False)[0] == 0.)
    for _ in range(20):
        assert np.max(abs(memory.choose(x,rng,True)[0]))<=1.0
