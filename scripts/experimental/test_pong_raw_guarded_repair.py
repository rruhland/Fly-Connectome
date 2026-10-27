import time,unittest
import numpy as np
import scipy.sparse as sp
from pong_raw_guarded_repair import fit

class GuardedRepairTest(unittest.TestCase):
    def test_lazy_matches_full_and_preserves_choices(self):
        rng=np.random.default_rng(72);W=rng.normal(size=(3,5));G=sp.csr_matrix(rng.normal(size=(20,5)));a=np.argmax(G@W.T,axis=1)
        guards=[dict(seed=1,group='first_left',X=G,actions=a)];X=sp.csr_matrix(rng.normal(size=(2,5)));b=(np.argmax(X@W.T,axis=1)+1)%3
        positive=[dict(seed=2,group='later_right',X=X,actions=b)];normal=dict(input_rms=2.,kappa=.03)
        delta,r=fit(W,positive,guards,normal,time.monotonic()+5,ridge=.01,batch=4)
        full,rf=fit(W,positive,guards,normal,time.monotonic()+5,ridge=.01,initial_rows=range(40))
        self.assertEqual(r['guard_action_changes'],0);self.assertLessEqual(r['final_objective'],r['zero_residual_objective'])
        self.assertAlmostEqual(r['final_objective'],rf['final_objective'],places=6);np.testing.assert_allclose(delta,full,atol=1e-4)

if __name__=='__main__':unittest.main()
