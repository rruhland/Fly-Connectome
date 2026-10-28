import time,unittest
import numpy as np
import scipy.sparse as sp
from pong_constrained_residual import CONTRAST
from pong_fixed_pairwise import FixedPairwise,FrozenPairwiseReadout
from pong_projected_reward import crossing_scale,GuardProjector


class ProjectedRewardTests(unittest.TestCase):
    def test_weighted_antithetic_crossing_scale(self):
        q=np.array([[1.,0.,0.],[2.,0.,0.]]);d=np.array([[0.,1.,0.],[0.,1.,0.]]);alpha=crossing_scale(q,np.array([0,0]),d,np.array([.8,.2]),.1);self.assertGreater(alpha,1);self.assertLess(alpha,2)
        self.assertEqual(np.argmax(q[0]+alpha*d[0]),1);self.assertEqual(np.argmax(q[0]-alpha*d[0]),0)

    def test_projection_keeps_nonzero_carried_skill_and_exact_ties(self):
        b=FixedPairwise(np.array([[0,1],[1,0]]),np.ones((2,2)),np.ones(2),np.array([.1,.2]),np.ones(3));W=np.zeros((3,2));X=sp.csr_matrix([[1.,0.],[0.,1.]]);Phi=b.batch(X);R0=np.zeros((3,3));R0[1]=Phi[0];p=FrozenPairwiseReadout(W,b,R0);aq=[p.choose(X.indices[X.indptr[i]:X.indptr[i+1]],X.data[X.indptr[i]:X.indptr[i+1]]) for i in range(2)];actions=np.array([v[0] for v in aq]);q=np.array([v[1] for v in aq]);project=GuardProjector(p,X,actions,q,np.eye(3),.1);center=-10*(CONTRAST.T@R0);R,r=project.project(center,time.monotonic()+5);self.assertTrue(r['verified']);self.assertEqual(r['original_deployed_rows_certified'],2)
        z=FrozenPairwiseReadout(W,b,R)
        for i,a in enumerate(actions):
            lo,hi=X.indptr[i:i+2];self.assertEqual(z.choose(X.indices[lo:hi],X.data[lo:hi])[0],a)
        self.assertGreater(np.linalg.norm(R-R0),0)


if __name__=='__main__':unittest.main()
