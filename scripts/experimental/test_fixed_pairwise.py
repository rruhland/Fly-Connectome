import tempfile,time,unittest
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from fly_connectome.control.policy import FrozenLinearPolicy
from pong_fixed_pairwise import FixedPairwise,FrozenPairwiseReadout,save_readout
from pong_fixed_rbf import load_readout
from run_rally_rbf_residual import fit_residual
from rally_action_equivalence import certify_episode


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

    def test_carried_nonlinear_reference_and_distinct_historical_source(self):
        b=self.basis(5);W=np.zeros((3,4));W[0,0]=.2;X=sp.csr_matrix([[.7,0.,-.5,0.]]);phi=b.batch(X)[0];R0=np.zeros((3,5));R0[1]=phi/(phi@phi);carried=FrozenPairwiseReadout(W,b,R0);self.assertEqual(carried.choose(X.indices,X.data)[0],1)
        current=dict(X=X,actions=np.array([1]),source_weights=W,source_readout=carried);R,r=fit_residual(W,b,[current],np.eye(5),1.,time.monotonic()+5,FrozenPairwiseReadout,carried);self.assertTrue(r['verified']);np.testing.assert_allclose(R,R0,atol=1e-12)
        historical=dict(X=X,actions=np.array([0]),source_weights=W);R,r=fit_residual(W,b,[historical],np.eye(5),1.,time.monotonic()+5,FrozenPairwiseReadout,carried);self.assertTrue(r['verified']);self.assertEqual(FrozenPairwiseReadout(W,b,R).choose(X.indices,X.data)[0],0)

    def test_nonlinear_source_receipt_requires_exact_readout(self):
        b=self.basis();W=np.zeros((3,4));R=np.zeros((3,7));R[1]=b.batch(sp.csr_matrix([[.7,0.,-.5,0.]]))[0];p=FrozenPairwiseReadout(W,b,R);X=sp.csr_matrix([[.7,0.,-.5,0.]]);a,q=p.choose(X.indices,X.data);config=dict(policy_kind='trained',action_hold_frames=1,episodes=1,action_seed=None);r=dict(complete=True,report=dict(complete=True),weights_sha256='bundle',model_kind=p.schema,config=config,feature_format='actual capped raw CSR',transitions=1,initial_cameras=1,records=[dict(action=a,q=q.tolist())]);kw=dict(source_weights_sha256='bundle',source_fingerprint={'v':1},candidate_fingerprint={'v':1},candidate_config=config,candidate_readout=p)
        self.assertFalse(certify_episode(r,X,W,W,**kw)['certified']);self.assertTrue(certify_episode(r,X,W,W,source_readout=p,**kw)['certified'])


if __name__=='__main__':unittest.main()
