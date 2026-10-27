import unittest
import numpy as np
import scipy.sparse as sp
from pong_rally_raw_margin import problem,objective,hessian_operator,fit
import time

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

    def test_active_rows_equal_dense_operator_and_warm_start(self):
        rng=np.random.default_rng(38);W=rng.normal(size=(3,5));X=sp.csr_matrix(rng.normal(size=(17,5)));a=np.argmax(X@W.T,axis=1)
        pos=[dict(seed=1,group='later_left',X=X,actions=(a+1)%3)];pres=[dict(seed=2,group='first_right',X=X,actions=a)]
        normal=dict(input_rms=2.,kappa=.1);data=problem(W,pos,pres,[],normal)
        p=rng.normal(size=10);value,g,h=objective(p,data,.01);H,_=hessian_operator(data,h,.01);v=rng.normal(size=10)
        U,c,old,required,w=data;z=U@v.reshape(2,5).T;dm=np.einsum('ni,nji->nj',z,c)
        expected=(U.T@np.sum((2*w*(h>0)*dm)[:,:,None]*c,axis=1)).T.reshape(-1)+.02*v
        np.testing.assert_array_equal(H@v,expected)
        d,report=fit(W,pos,pres,[],normal,deadline=time.monotonic()+3,ridge=.01)
        d2,r2=fit(W,pos,pres,[],normal,deadline=time.monotonic()+3,ridge=.01,initial_delta=d)
        self.assertTrue(r2['warm_start']);self.assertLessEqual(r2['final_objective'],report['final_objective']+1e-12)

if __name__=='__main__':unittest.main()
