import unittest
import numpy as np
from policy_search_math import spatial_basis
from pong_spatial_extension import extra_transform,map_split


class SpatialExtensionTests(unittest.TestCase):
    def test_new_modes_and_exact_zero_added_mapping(self):
        layout={'current':(0,256),'metadata':(256,264)};rng=np.random.default_rng(917)
        old,_=spatial_basis(layout,264,3);raw=rng.normal(size=(80,264));raw[:,259]=1.
        extra,info=extra_transform(raw,layout,264)
        self.assertEqual(extra.shape,(16,264));self.assertEqual(info['extra_rows'],16)
        theta=rng.normal(size=(3,len(old)));extended=np.pad(theta,((0,0),(0,len(extra))))
        self.assertTrue(np.array_equal(map_split(extended,old,extra),theta@old))
        extended[:,len(old):]=rng.normal(size=(3,len(extra)))
        features=raw@np.vstack([old,extra]).T
        np.testing.assert_allclose(raw@map_split(extended,old,extra).T,features@extended.T,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(raw@extra.T,(raw@extra.T)-(raw@extra.T).mean(0),atol=1e-12)


if __name__=='__main__':unittest.main()
