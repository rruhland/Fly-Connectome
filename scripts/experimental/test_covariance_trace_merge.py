import time,unittest
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from pong_covariance_trace_merge import fit


class CovarianceTraceMergeTest(unittest.TestCase):
    def test_correlated_metric_retains_whole_hard_target(self):
        reference=np.array([[0.,-.1],[0.,0.],[.1,.1]])
        X=sp.csr_matrix([[1.,0.],[0.,1.]])
        case=dict(X=X,actions=np.array([0,2]),source_weights=reference,interventions=[dict(row=0,action=0,source_action=2,reward_verified=True)])
        G=np.array([[2.,.8],[.8,1.]])
        candidate,report=fit(reference,case,np.array([2.,.5]),la.cholesky(G,lower=True),.03,time.monotonic()+3)
        self.assertTrue(report['verified']);self.assertFalse(report['old_actions_hard_preserved']);np.testing.assert_array_equal(np.argmax(candidate,axis=0),[0,2])
        delta=(candidate-reference)*np.array([2.,.5]);self.assertAlmostEqual(report['whitened_parameter_cost'],float(np.sum((delta@G)*delta)),places=8)
        self.assertLessEqual(report['maximum_actual_target_logit_violation'],1e-9)

    def test_multiple_complete_cases_share_the_same_score_metric(self):
        reference=np.array([[0.,-.1],[0.,0.],[.1,.1]])
        cases=[dict(X=sp.csr_matrix([[1.,0.]]),actions=np.array([0]),source_weights=reference,interventions=[dict(row=0,action=0,source_action=2,reward_verified=True)]),dict(X=sp.csr_matrix([[0.,1.]]),actions=np.array([2]),source_weights=reference)]
        candidate,report=fit(reference,cases,np.array([2.,.5]),la.cholesky(np.array([[2.,.8],[.8,1.]]),lower=True),.03,time.monotonic()+3)
        self.assertTrue(report['verified']);self.assertEqual(report['hard_cases'],2);self.assertEqual(report['verified_exact_target_greedy_rows'],2);np.testing.assert_array_equal(np.argmax(candidate,axis=0),[0,2])


if __name__=='__main__':unittest.main()
