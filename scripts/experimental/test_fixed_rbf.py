import time,unittest
import numpy as np
import scipy.sparse as sp
from fly_connectome.control.policy import FrozenLinearPolicy
from pong_fixed_rbf import FixedRBF,FrozenRBFReadout
from run_rally_rbf_residual import fit_residual


class FixedRBFTest(unittest.TestCase):
    def test_sparse_deployment_matches_batch_features(self):
        centers=np.array([[.3,-.7],[1.2,.1]]);basis=FixedRBF(centers,np.array([2.,.5]),.8);X=sp.csr_matrix([[0.,.2],[1.,-.3],[0.,0.]])
        for i in range(len(X.indptr)-1):
            lo,hi=X.indptr[i:i+2];np.testing.assert_allclose(basis.features(X.indices[lo:hi],X.data[lo:hi]),basis.batch(X)[i],rtol=0,atol=1e-15)

    def test_zero_residual_delegates_exact_q_and_legal_ties(self):
        W=np.zeros((3,2));basis=FixedRBF(np.eye(2),np.ones(2),1.);readout=FrozenRBFReadout(W,basis,np.zeros((3,2)));indices=np.array([0,1]);values=np.array([.3,.7]);legal=np.array([2,0,1]);a,q=readout.choose(indices,values,legal);old,oldq=FrozenLinearPolicy(W).choose(indices,values,legal);self.assertEqual(a,old);np.testing.assert_array_equal(q,oldq);self.assertEqual(a,2)
        with self.assertRaises(ValueError):readout.residual[0,0]=1

    def test_only_residual_weights_fit_with_original_row_certificate(self):
        W=np.array([[0.,-.1],[0.,0.],[.1,.1]]);basis=FixedRBF(np.column_stack((np.linspace(-1,1,64),np.zeros(64))),np.ones(2),.2);case=dict(X=sp.csr_matrix([[1.,0.],[0.,1.]]),actions=np.array([0,2]),source_weights=W,interventions=[dict(row=0,action=0,source_action=2,reward_verified=True)]);original=W.copy();R,report=fit_residual(W,basis,[case],np.eye(64),.03,time.monotonic()+3)
        self.assertTrue(report['verified']);self.assertEqual(report['original_deployed_choices_certified'],2);np.testing.assert_array_equal(W,original);policy=FrozenRBFReadout(W,basis,R)
        for i,a in enumerate(case['actions']):
            lo,hi=case['X'].indptr[i:i+2];self.assertEqual(policy.choose(case['X'].indices[lo:hi],case['X'].data[lo:hi])[0],a)


if __name__=='__main__':unittest.main()
