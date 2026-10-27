import unittest
import numpy as np
from pong_rally_imitation import case_weights,fit
class WeightsTests(unittest.TestCase):
 def test_arbitrary_groups_episodes_segments_and_row_duplication(self):
  cases=[dict(group='first',seed=1,X=np.zeros((2,2))),dict(group='later',seed=2,X=np.zeros((3,2))),dict(group='later',seed=2,X=np.zeros((1,2))),dict(group='later',seed=3,X=np.zeros((1,2))),dict(group='third',seed=4,X=np.zeros((1,2)))]
  w=case_weights(cases);self.assertAlmostEqual(w.sum(),1.);self.assertAlmostEqual(w[:2].sum(),1/3);self.assertAlmostEqual(w[2:7].sum(),1/3);self.assertAlmostEqual(w[-1],1/3)
 def test_finite_rowcentered_fit_with_soft_preservation(self):
  theta=np.array([[.01,0.],[0.,.01],[-.01,-.01]])
  p=[dict(group='later',seed=1,X=np.array([[1.,0.]]),actions=np.array([1]))]
  s=[dict(group='first',seed=2,X=np.array([[0.,1.]]),actions=np.array([1]))]
  d,r=fit(theta,p,s,[],dict(input_rms=1.,kappa=.05))
  self.assertTrue(np.isfinite(d).all());self.assertLess(np.max(abs(d.sum(0))),1e-12);self.assertEqual(r['positive']['weighted_requested_action_fraction'],1.)
if __name__=='__main__':unittest.main()
