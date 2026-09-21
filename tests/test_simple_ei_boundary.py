import sys
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('scipy')
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from simple_ei_boundary import features, branches, quality


def test_local_coordinates_preserve_inhibitory_magnitude_and_zero_ratio():
    x=np.array([[0.,0.,0.],[1.,2.,-1.],[-1.,1.,-2.]])
    np.testing.assert_allclose(features(x,'ratio').ravel(),[0.,1/3,2/3])
    np.testing.assert_array_equal(features(x,'E_H'),[[0,0],[2,1],[1,2]])
    np.testing.assert_array_equal(features(x,'product').ravel(),[0,2,2])


def test_boundary_requires_both_coordinates_and_counts_all_quiet_calls():
    z=np.array([[0.,0.],[2.,1.],[1.,2.],[2.,2.]])
    bounds=np.array([[[1.5,.5],[2.5,1.5]],[[.5,1.5],[1.5,2.5]]])
    calls=branches(z,bounds)
    np.testing.assert_array_equal(calls,[[False,False],[True,False],[False,True],[False,False]])
    labels=np.array([0,-1,1,0])
    assert quality(calls.any(1),labels)==(1.,2.,-0.)
    assert quality(np.ones(4,dtype=bool),labels) is None
