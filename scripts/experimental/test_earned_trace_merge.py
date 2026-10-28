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

if __name__=='__main__':unittest.main()
