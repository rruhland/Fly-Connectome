import unittest
import numpy as np
import scipy.sparse as sp
from pong_rally_raw_margin import problem,objective,hessian_operator

class RawMarginTest(unittest.TestCase):
    def test_analytic_gradient_and_hessian_product(self):
        rng=np.random.default_rng(9173);W=rng.normal(size=(3,7));X=sp.csr_matrix(rng.normal(size=(12,7)));actions=np.argmax(X@W.T,axis=1)
        preserve=[dict(seed=3,group='later_left',X=X,actions=actions)]
        positive=[dict(seed=4,group='later_right',X=X,actions=(actions+1)%3)]
        data=problem(W,positive,preserve,[],dict(input_rms=2.,kappa=.13));p=rng.normal(size=14)*.1;v=rng.normal(size=14);eps=1e-6
        value,g,h=objective(p,data,ridge=.07);H,_=hessian_operator(data,h,.07)
        diff=(objective(p+eps*v,data,.07,False)-objective(p-eps*v,data,.07,False))/(2*eps)
        self.assertAlmostEqual(diff,float(g@v),places=6)
        np.testing.assert_allclose(H@v,(objective(p+eps*v,data,.07)[1]-objective(p-eps*v,data,.07)[1])/(2*eps),rtol=1e-6,atol=1e-6)

if __name__=='__main__':unittest.main()
