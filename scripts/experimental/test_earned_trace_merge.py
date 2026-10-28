import time,unittest
import numpy as np
import scipy.sparse as sp
from pong_earned_trace_merge import merge

class EarnedMergeTest(unittest.TestCase):
    def test_verifies_merge_without_feasible_anchor(self):
        ref=np.array([[-.1,-1.],[.1,0.],[0.,1.]])
        source=np.array([[1.,-1.],[0.,0.],[-1.,1.]])
        cases=[dict(X=sp.eye(2,format='csr'),actions=np.array([0,2]),source_weights=source)]
        candidate,r=merge(ref,cases,np.array([2.,.5]),.03,time.monotonic()+3)
        self.assertTrue(r['verified']);self.assertFalse(r['known_feasible_anchor']);np.testing.assert_array_equal(np.argmax(candidate,axis=0),[0,2])

    def test_incompatible_constraints_do_not_return_a_fallback(self):
        ref=np.zeros((3,1));X=sp.csr_matrix([[1.]])
        cases=[dict(X=X,actions=np.array([0]),source_weights=np.array([[1.],[0.],[-1.]])),dict(X=X,actions=np.array([1]),source_weights=np.array([[0.],[1.],[-1.]]))]
        candidate,r=merge(ref,cases,np.ones(1),.03,time.monotonic()+3)
        self.assertIsNone(candidate);self.assertFalse(r['verified']);self.assertFalse(r['numerical_fallback'])

    def test_declared_reward_verified_override_gets_positive_margin(self):
        ref=np.array([[0.],[0.],[.1]])
        case=dict(seed=7,X=sp.csr_matrix([[1.]]),actions=np.array([0]),source_weights=ref,interventions=[dict(row=0,action=0,source_action=2,reward_verified=True)])
        candidate,r=merge(ref,[case],np.ones(1),.03,time.monotonic()+3)
        self.assertTrue(r['verified']);self.assertEqual(r['verified_exact_greedy_rows'],1)
        self.assertGreater(candidate[0,0]-candidate[2,0],2.9e-6)
        self.assertEqual(r['intervention_rows'][0]['source_action'],2)

    def test_undeclared_non_greedy_action_is_rejected(self):
        ref=np.array([[0.],[0.],[.1]])
        case=dict(X=sp.csr_matrix([[1.]]),actions=np.array([0]),source_weights=ref)
        with self.assertRaisesRegex(AssertionError,'undeclared non-greedy'):
            merge(ref,[case],np.ones(1),.03,time.monotonic()+3)

if __name__=='__main__':unittest.main()
