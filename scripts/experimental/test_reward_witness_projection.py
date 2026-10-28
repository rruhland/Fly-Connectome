import unittest,time
import numpy as np
import scipy.sparse as sp
from pong_reward_witness_projection import project

class ProjectionTest(unittest.TestCase):
    def test_nearest_action_cone_matches_closed_form(self):
        reference=np.array([[-.1,-1.],[.1,0.],[0.,1.]])
        feasible=np.array([[1.,-1.],[0.,0.],[-1.,1.]])
        result,report=project(reference,feasible,sp.eye(2,format='csr'),np.array([0,2]),np.array([2.,.5]),100.,time.monotonic()+3)
        np.testing.assert_allclose(result[:,0],[.0066666667,-.0033333333,-.0033333333],atol=1e-6)
        np.testing.assert_allclose(result[:,1],reference[:,1],atol=1e-6)
        self.assertEqual(report['whole_witness_action_changes'],[]);self.assertLess(report['projection_cost'],report['known_feasible_cost'])

    def test_pair_does_not_order_an_unmeasured_third_action(self):
        reference=np.array([[1.,0.],[0.,2.5],[-1.,3.]])
        feasible=np.array([[1.,3.],[0.,2.5],[-1.,0.]])
        pair=dict(X=sp.csr_matrix([[0.,1.]]),good=0,bad=2)
        result,report=project(reference,feasible,sp.csr_matrix([[1.,0.]]),np.array([0]),np.ones(2),100.,time.monotonic()+3,[pair])
        np.testing.assert_allclose(result[:,1],[1.505,2.5,1.495],atol=1e-6)
        self.assertEqual(int(np.argmax(result[:,1])),1);self.assertEqual(report['extra_measured_pairs'],1)

if __name__=='__main__':unittest.main()
