import unittest
import numpy as np
from pong_action_credit import preferences,fit_preferences

class CreditTests(unittest.TestCase):
    def test_no_fabricated_ties(self):
        self.assertEqual(preferences([1,1,1],[4,2,1]),[])
        self.assertEqual(preferences([0,0,0],[4,2,1]),[])
        self.assertEqual(preferences([1,1,0],[4,2,3]),[(0,2)])
        self.assertEqual(preferences([1,1,0],[2,4,3]),[(1,2)])
    def test_no_contrasts_no_fit(self):
        t=np.zeros((3,2));s=[dict(seed=1,phase='left',x=[1.,0.],outcomes=[0,0,0],base_action=0)]
        delta,report=fit_preferences(t,s,np.eye(2),np.ones(2)/2)
        self.assertIsNone(delta)
    def test_pairwise_fit_and_dose(self):
        theta=np.array([[.1,0.],[0.,.1],[-.1,-.1]])
        states=[dict(seed=1,phase='left',x=[1.,0.],outcomes=[0,1,1],base_action=0),dict(seed=2,phase='right',x=[0.,1.],outcomes=[1,0,0],base_action=1)]
        delta,r=fit_preferences(theta,states,np.eye(2),np.ones(2)/2,maximum_rms=1.,anchor_weight=0.,ridge=1e-8)
        self.assertLess(np.max(abs(delta.sum(0))),1e-12)
        self.assertEqual(r['raw']['repair_states_greedy_success'],2)
        self.assertGreater((theta+delta)[1,0],(theta+delta)[0,0])
        self.assertGreater((theta+delta)[0,1],(theta+delta)[1,1])
        capped,rc=fit_preferences(theta,states,np.eye(2),np.ones(2)/2,maximum_rms=.001,anchor_weight=0.,ridge=1e-8)
        self.assertLessEqual(rc['applied_legal_contrast_rms'],.0010000001)
        self.assertGreater(rc['raw']['repair_states_greedy_success'],rc['applied']['repair_states_greedy_success'])
    def test_satisfied_success_anchor_is_zero(self):
        theta=np.array([[1.,0.],[0.,1.],[-1.,-1.]])
        states=[dict(seed=1,phase='left',x=[1.,0.],outcomes=[1,0,0],base_action=0),dict(seed=2,phase='right',x=[0.,1.],outcomes=[0,1,0],base_action=1)]
        delta,r=fit_preferences(theta,states,np.eye(2),np.ones(2)/2)
        self.assertTrue(np.array_equal(delta,np.zeros_like(delta)))

if __name__=='__main__':unittest.main()
