import unittest
import numpy as np
from pong_rally_margin import problem,objective,active_hessian,fit
class MarginTests(unittest.TestCase):
 def data(self):
  theta=np.array([[.1,0.],[0.,.1],[-.1,-.1]])
  p=[dict(group='later',seed=1,X=np.array([[1.,0.],[.7,.2]]),actions=np.array([1,1]))]
  s=[dict(group='first',seed=2,X=np.array([[0.,1.]]),actions=np.array([1]))]
  return theta,p,s
 def test_gradient_hessian_without_autodiff(self):
  t,p,s=self.data();data=problem(t,p,s,[],dict(input_rms=1.,kappa=.05));x=np.array([.02,-.01,.03,.01]);f,g,h=objective(x,data);H=active_hessian(data,h);eps=1e-6
  numerical=[];columns=[]
  for i in range(4):
   e=np.eye(4)[i]*eps;numerical.append((objective(x+e,data,need_gradient=False)-objective(x-e,data,need_gradient=False))/(2*eps));columns.append((objective(x+e,data)[1]-objective(x-e,data)[1])/(2*eps))
  self.assertTrue(np.allclose(g,numerical,atol=1e-8));self.assertTrue(np.allclose(H,np.array(columns).T,atol=1e-7))
 def test_softfit_descends_and_prefers_rewarded_action(self):
  t,p,s=self.data();delta,r=fit(t,p,s,[],dict(input_rms=1.,kappa=.05));self.assertLess(r['final_objective'],r['zero_residual_objective']);self.assertGreater(r['positive']['weighted_requested_action_fraction'],0.9);self.assertLess(np.max(abs(delta.sum(0))),1e-12)
if __name__=='__main__':unittest.main()
