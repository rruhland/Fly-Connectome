import tempfile,time,unittest
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from fly_connectome.control.policy import FrozenLinearPolicy
from pong_fixed_pairwise import FixedPairwise,FrozenPairwiseReadout,save_readout
from pong_fixed_rbf import load_readout
from run_rally_rbf_residual import fit_residual


class PairwiseTests(unittest.TestCase):
    def basis(self,m=7):
        rng=np.random.default_rng(13);return FixedPairwise(rng.integers(m,size=(2,4)),rng.choice([-1,1],size=(2,4)),np.array([.4,2.,3.,.2]),np.array([.1,-.3,.2,.6]),np.arange(1,m+1)/3)

    def test_direct_scaled_centered_products_and_batch(self):
        b=self.basis();x=np.array([.7,0.,-.5,0.]);z=x/b.scale-b.mean;expected=np.zeros(b.output_dim)
        for i in range(4):
            for j in range(4):expected[(b.hashes[0,i]+b.hashes[1,j])%b.output_dim]+=b.signs[0,i]*b.signs[1,j]*z[i]*z[j]
        expected/=b.rms;ix=np.flatnonzero(x);np.testing.assert_allclose(b.features(ix,x[ix]),expected,atol=1e-14);np.testing.assert_allclose(b.batch(sp.csr_matrix([x]))[0],expected,atol=1e-14)

    def test_zero_tie_and_bundle(self):
        b=self.basis();W=np.zeros((3,4));p=FrozenPairwiseReadout(W,b,np.zeros((3,7)));ix=np.array([0]);x=np.array([.7]);legal=np.array([2,0,1]);a,q=p.choose(ix,x,legal);aa,qq=FrozenLinearPolicy(W).choose(ix,x,legal);self.assertEqual(a,aa);np.testing.assert_array_equal(q,qq)
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/'bundle.npz';save_readout(f,W,b,p.residual);loaded=load_readout(f);self.assertEqual(loaded.schema,p.schema);np.testing.assert_array_equal(loaded.choose(ix,x,legal)[1],q);np.testing.assert_array_equal(loaded.basis.features(ix,x),b.features(ix,x))

    def test_non64_fit_dimensions_and_deployed_certificate(self):
        b=self.basis(5);W=np.zeros((3,4));X=sp.csr_matrix([[.7,0.,-.5,0.]]);case=dict(X=X,actions=np.array([1]),source_weights=W,interventions=[dict(row=0,reward_verified=True,action=1,source_action=0)])
        R,r=fit_residual(W,b,[case],np.eye(5),1.,time.monotonic()+5,FrozenPairwiseReadout);self.assertTrue(r['verified']);self.assertEqual(R.shape,(3,5));self.assertEqual(r['contrast_parameters'],10);self.assertEqual(FrozenPairwiseReadout(W,b,R).choose(X.indices,X.data)[0],1)


if __name__=='__main__':unittest.main()
