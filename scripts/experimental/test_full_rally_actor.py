import unittest
import numpy as np
import scipy.linalg as la
from pong_constrained_residual import CONTRAST
from pong_full_rally_actor import probabilities,calibrate_temperature,score,paired_gradient,capped_step


class FullActorTests(unittest.TestCase):
    def test_score_matches_local_finite_difference(self):
        rng=np.random.default_rng(19);theta=rng.normal(size=(2,5));f=rng.normal(size=5);temperature=.7;action=1;q=CONTRAST@theta@f;p=probabilities(q,temperature);analytic=score(action,p,f,temperature);numeric=np.zeros_like(theta);epsilon=1e-6
        for i in range(2):
            for j in range(5):
                plus=theta.copy();minus=theta.copy();plus[i,j]+=epsilon;minus[i,j]-=epsilon;numeric[i,j]=(np.log(probabilities(CONTRAST@plus@f,temperature)[action])-np.log(probabilities(CONTRAST@minus@f,temperature)[action]))/(2*epsilon)
        np.testing.assert_allclose(analytic,numeric,atol=1e-9)

    def test_all_hits_credit_earlier_actions_only(self):
        # Hit at decision0 and decision2: the final action cannot receive past reward.
        scores=np.array([[[1.,2.]],[[3.,4.]],[[5.,6.]]]);eligibility=np.zeros((1,2));credit=eligibility.copy()
        for s,hit in zip(scores,[1,0,1]):eligibility+=s;credit+=hit*eligibility
        np.testing.assert_array_equal(credit,2*scores[0]+scores[1]+scores[2]);self.assertFalse(np.array_equal(credit,2*eligibility))
        e=[dict(seed=s,complete=True,hits=r,credit=c,eligibility=v) for s,r,c,v in [(1,2,credit,eligibility),(1,2,credit+1,eligibility-1),(2,0,np.zeros((1,2)),np.ones((1,2))),(2,1,np.ones((1,2)),np.ones((1,2)))]]
        expected=(e[0]['credit']-2*e[0]['eligibility']+e[1]['credit']-2*e[1]['eligibility']+e[2]['credit']-e[2]['eligibility']+e[3]['credit'])/4
        np.testing.assert_array_equal(paired_gradient(e),expected)
        self.assertTrue(np.any(e[0]['credit']+e[1]['credit']-2*(e[0]['eligibility']+e[1]['eligibility'])))
        e[0]['complete']=False
        with self.assertRaises(AssertionError):paired_gradient(e)

    def test_unattainable_temperature_rejected(self):
        with self.assertRaises(ValueError):calibrate_temperature(np.zeros((10,3)))

    def test_temperature_and_two_fixed_caps(self):
        rng=np.random.default_rng(41);F=rng.normal(size=(128,8));theta=rng.normal(size=(2,8));q=(F@theta.T)@CONTRAST.T;temperature,rate=calibrate_temperature(q);self.assertAlmostEqual(rate,.05,places=12);G=rng.normal(size=(8,8));G=G@G.T+.1*np.eye(8);L=la.cholesky(G,lower=True);gradient=rng.normal(size=(2,8));initial=float(np.sqrt(np.mean(q*q)));delta,report=capped_step(gradient,L,F,q,temperature,initial);actual=(F@delta.T)@CONTRAST.T
        self.assertLessEqual(np.sqrt(np.mean(actual*actual)),.01*initial+1e-12);self.assertLessEqual(report['probability_TV'],.01+1e-12);self.assertTrue(report['nonzero_update']);np.testing.assert_allclose(CONTRAST.T@CONTRAST,np.eye(2),atol=1e-15)
        zero,report=capped_step(np.zeros_like(gradient),L,F,q,temperature,initial);self.assertFalse(report['nonzero_update']);np.testing.assert_array_equal(zero,np.zeros_like(zero))


if __name__=='__main__':unittest.main()
