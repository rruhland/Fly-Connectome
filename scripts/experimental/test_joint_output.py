import tempfile,time,unittest
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import osqp
from fly_connectome.control.policy import FrozenLinearPolicy
from pong_constrained_residual import CONTRAST
from pong_fixed_pairwise import FixedPairwise,FrozenPairwiseReadout,save_readout
from pong_fixed_rbf import load_readout
from pong_joint_output import fit,undo,metric


class JointOutputTests(unittest.TestCase):
    def fixture(self):
        b=FixedPairwise(np.array([[0,1],[1,0]]),np.ones((2,2)),np.ones(2),np.zeros(2),np.ones(3));W=np.zeros((3,2));R=np.zeros((3,3));R[1]=.8;return FrozenPairwiseReadout(W,b,R),sp.eye(2,format='csr')

    def test_optimizer_undo_prediction_and_zero_identity(self):
        p,X=self.fixture();den=np.array([2.,.5]);pd=1.3;theta=np.arange(10,dtype=float).reshape(2,5)/100;W,R=undo(p,theta,den,pd);F=np.hstack((X.toarray()/den,p.basis.batch(X)/pd));expected=F@theta.T@CONTRAST.T;actual=X.toarray()@(W-p.base.weights).T+p.basis.batch(X)@(R-p.residual).T;np.testing.assert_allclose(actual,expected,atol=1e-15)
        W0,R0=undo(p,np.zeros_like(theta),den,pd);z=FrozenPairwiseReadout(W0,p.basis,R0)
        for i in range(2):
            lo,hi=X.indptr[i:i+2];np.testing.assert_array_equal(z.choose(X.indices[lo:hi],X.data[lo:hi])[1],p.choose(X.indices[lo:hi],X.data[lo:hi])[1])

    def test_joint_dual_matches_independent_dense_primal_and_bundle(self):
        p,X=self.fixture();source=FrozenLinearPolicy(np.array([[.4,-.5],[0.,0.],[-.4,.5]]));actions=np.array([0,2]);case=dict(X=X,actions=actions,source_readout=source,source_policy_sha256='historical',interventions=[]);den=np.array([2.,.5]);pd=1.3;G=np.eye(5);G[0,3]=G[3,0]=.3;G[1,3]=G[3,1]=-.2;result,r=fit(p,[case],den,pd,la.cholesky(G,lower=True),.1,time.monotonic()+5,'carry');self.assertTrue(r['verified']);W,R,theta=result;self.assertGreater(np.linalg.norm(W-p.base.weights),0);self.assertTrue(r['source_policies_distinct_from_reference_cases'])
        F=np.hstack((X.toarray()/den,p.basis.batch(X)/pd));A=[];lower=[]
        for i,a in enumerate(actions):
            lo,hi=X.indptr[i:i+2];old=p.choose(X.indices[lo:hi],X.data[lo:hi])[1];q=source.choose(X.indices[lo:hi],X.data[lo:hi])[1]
            for b in range(3):
                if b==a:continue
                c=CONTRAST[a]-CONTRAST[b];A.append(np.r_[c[0]*F[i],c[1]*F[i]]);lower.append(min(.1*1e-4,.5*(q[a]-q[b]))-(old[a]-old[b]))
        control=osqp.OSQP();control.setup(P=sp.block_diag((sp.csc_matrix(G),sp.csc_matrix(G)),format='csc'),q=np.zeros(10),A=sp.csc_matrix(A),l=np.array(lower),u=np.full(4,np.inf),verbose=False,eps_abs=1e-11,eps_rel=1e-11,polishing=True);answer=control.solve(raise_error=False);np.testing.assert_allclose(theta.ravel(),answer.x,atol=1e-8)
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);np.save(t/'proposal.npy',W);save_readout(t/'bundle.npz',W,p.basis,R);loaded=load_readout(t/'bundle.npz');np.testing.assert_array_equal(loaded.base.weights,np.load(t/'proposal.npy'));np.testing.assert_array_equal(p.base.weights,np.zeros((3,2)))
            for i,a in enumerate(actions):
                lo,hi=X.indptr[i:i+2];self.assertEqual(loaded.choose(X.indices[lo:hi],X.data[lo:hi])[0],a)

    def test_metric_normalizes_optimizer_blocks_only(self):
        p,X=self.fixture();before=p.basis.frozen_parameters if hasattr(p.basis,'frozen_parameters') else p.basis.rms.copy();G,den,pd,r=metric([X],p.basis,np.array([2.,.5]));self.assertAlmostEqual(np.trace(G)/5,1.01);np.testing.assert_array_equal(p.basis.rms,before);self.assertTrue(r['crossblock_correlations_included'])

    def test_already_satisfied_contract_has_zero_update(self):
        p,X=self.fixture();c=dict(X=X,actions=np.array([1,1]),source_readout=p,source_policy_sha256='carry',interventions=[]);result,r=fit(p,[c],np.ones(2),1.,np.eye(5),.1,time.monotonic()+5,'carry');self.assertTrue(r['verified']);np.testing.assert_allclose(result[0],p.base.weights,atol=1e-14);np.testing.assert_allclose(result[1],p.residual,atol=1e-14)


if __name__=='__main__':unittest.main()
