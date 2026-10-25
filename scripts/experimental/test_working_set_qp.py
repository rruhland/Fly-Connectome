import numpy as np
import scipy.sparse as sp
import osqp
from pong_working_set_qp import solve_qp


def test_full_feasibility_and_dual_certificate_match_full_qp():
    P=np.array([[2.,.3],[.3,3.]]);q=np.array([-5.,-8.]);A=np.array([[-1.,0.],[0.,-1.],[1.,1.],[1.,0.],[0.,1.]]);l=np.array([-1.,-1.,0.,-4.,-4.])
    x,info=solve_qp(P,q,A,l,batch=1)
    full=osqp.OSQP();full.setup(P=sp.csc_matrix(P),q=q,A=sp.csc_matrix(A),l=l,u=np.full(len(l),np.inf),verbose=False,eps_abs=1e-10,eps_rel=1e-10,polishing=True)
    answer=full.solve(raise_error=False)
    assert np.allclose(x,answer.x,atol=1e-7) and info['maximum_full_violation']<1e-7 and abs(info['duality_gap'])<1e-7
    assert info['working_rows']<len(l) and info['rounds']>=2


def test_unconstrained_optimum_and_no_violations():
    x,info=solve_qp(np.eye(2),np.array([-1.,-2.]),np.array([[1.,0.],[0.,1.]]),np.array([-1.,-1.]))
    assert np.allclose(x,[1.,2.]) and info['maximum_full_violation']==0 and info['working_rows']==0
