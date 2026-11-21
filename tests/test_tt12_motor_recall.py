import numpy as np
import pytest

from scripts.experimental.tt6_body_control import CameraBody
from scripts.experimental.tt12_motor_recall import rewire


def test_signed_port_relabeling_preserves_quadratic_pose_and_response():
    rng = np.random.default_rng(4793)
    n = 5
    fit = CameraBody(rng.random((1+n+n*(n+1)//2,4)),rng.random(n),
                     rng.random((n,14)),0.,100,100,True)
    order, signs = np.array([2,0,4,3,1]), np.array([-1,1,1,-1,1])
    new = rewire(fit,order,signs)
    ports = rng.normal(size=(14,n))
    old = np.zeros_like(ports)
    old[:,order] = ports * signs
    np.testing.assert_allclose(new.pose(ports),fit.pose(old))
    np.testing.assert_array_equal(new.kernels,fit.kernels[order])
    np.testing.assert_array_equal(new.lags,fit.lags[order])
    with pytest.raises(ValueError):
        rewire(fit,[1]*n,signs)
