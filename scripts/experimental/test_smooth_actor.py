import unittest
from types import SimpleNamespace
import numpy as np
from fly_connectome.vision import VisualStateEncoder
from pong_smooth_actor import correction_basis,projected_metric,mapped_weights
from pong_full_rally_actor import probabilities,score
from pong_constrained_residual import CONTRAST


class SmoothTests(unittest.TestCase):
    def test_basis_spatial_only_and_orthonormal(self):
        encoder=VisualStateEncoder(64,64,.02);B=correction_basis(encoder.layout,encoder.dimension);self.assertEqual(B.shape,(198,4389));self.assertFalse(np.any(B[:,3840:3848]));self.assertFalse(np.any(B[:,4381:]));self.assertTrue(np.any(B[:,4373:4381]));np.testing.assert_allclose(B@B.T,np.eye(198),atol=2e-15);delta=CONTRAST@np.ones((2,198))@B;self.assertTrue(np.any(delta[:,4373:4381]));self.assertFalse(np.any(delta[:,3840:3848]));self.assertFalse(np.any(delta[:,4381:]))

    def test_projection_metric_and_score_are_actual_mapped_coordinates(self):
        rng=np.random.default_rng(81);d=9;m=4;p=3;B=np.linalg.qr(rng.normal(size=(d,m)))[0].T;den=np.exp(rng.normal(size=d));A=rng.normal(size=(d+p,d+p));G=A@A.T+.01*np.eye(d+p);H=projected_metric(G,B,den);c=rng.normal(size=(2,m));joint=np.column_stack((c@(B*den),np.zeros((2,p))));self.assertAlmostEqual(float(np.sum((c@H)*c)),float(np.sum((joint@G)*joint)),places=9);self.assertFalse(np.allclose(H,B@G[:d,:d]@B.T))
        X=rng.normal(size=(16,d));W=rng.normal(size=(3,d));reference=SimpleNamespace(base=SimpleNamespace(weights=W));mapped=mapped_weights(reference,c,B);np.testing.assert_allclose(X@(mapped-W).T,(X@B.T@c.T)@CONTRAST.T,atol=1e-14);np.testing.assert_array_equal(mapped_weights(reference,np.zeros_like(c),B),W)
        x=X[0];temp=.8;q=W@x+CONTRAST@c@(B@x);a=2;actual=score(a,probabilities(q,temp),B@x,temp);eps=1e-6;numeric=np.zeros_like(c)
        for i in range(2):
            for j in range(m):
                plus=c.copy();minus=c.copy();plus[i,j]+=eps;minus[i,j]-=eps;qp=mapped_weights(reference,plus,B)@x;qm=mapped_weights(reference,minus,B)@x;numeric[i,j]=(np.log(probabilities(qp,temp)[a])-np.log(probabilities(qm,temp)[a]))/(2*eps)
        np.testing.assert_allclose(actual,numeric,atol=2e-9)


if __name__=='__main__':unittest.main()
