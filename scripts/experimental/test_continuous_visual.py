import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from pong_continuous_visual import canonical_scene,descriptors,fit_basis,ContinuousReadout,save_readout,save_scenes,load_scenes
from pong_fixed_pairwise import FixedPairwise,FrozenPairwiseReadout
from pong_fixed_rbf import load_readout
from pong_full_rally_actor import probabilities,score
from pong_constrained_residual import CONTRAST
from rally_action_equivalence import certify_episode


def calibration():
    rng=np.random.default_rng(10);return [[np.column_stack((rng.normal(size=(3,4)),np.ones((3,2)))) for _ in range(8)] for _ in range(2)]


class ContinuousTests(unittest.TestCase):
    def setUp(self):self.basis,self.F,self.G,self.weights,self.report=fit_basis(calibration())

    def test_permutation_ID_invariance_empty_unknown_signed_pairs(self):
        entities=[dict(id=i,normalized_position=np.array([.01*i,.02*i]),observed_velocity=np.array([.1*i,-.2*i]),observed=True,velocity_known=True,private_role='ignored') for i in (1,2,3)];a=canonical_scene(dict(entities=entities));b=canonical_scene(dict(entities=[dict(e,id=100-e['id']) for e in entities[::-1]]));np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(self.basis.features(a),self.basis.features(b));np.testing.assert_array_equal(self.basis.features(np.empty((0,6))),np.zeros(256));single=np.array([[0,0,0,0,1,0.]]);np.testing.assert_array_equal(self.basis.features(single)[128:],np.zeros(128));_,pairs=descriptors(a[:2]);np.testing.assert_array_equal(pairs[0,:4],a[0,:4]);np.testing.assert_allclose(pairs[0,4:8],a[1,:4]-a[0,:4]);self.assertEqual(len(pairs),2)

    def test_subbin_and_association_not_collapsed(self):
        a=np.array([[.01,.01,.2,-.1,1,1],[.02,.02,-.2,.1,1,1]]);b=a.copy();b[:,:4][:,2:]=a[::-1,2:4];np.testing.assert_allclose(a[:,:4].sum(0),b[:,:4].sum(0));self.assertGreater(np.linalg.norm(self.basis.features(a)-self.basis.features(b)),1e-7);c=a.copy();c[0,0]+=.001;self.assertGreater(np.linalg.norm(self.basis.features(a)-self.basis.features(c)),1e-7)

    def test_weighting_floors_metric_and_batch_features(self):
        episodes=calibration();basis,F,G,w,report=fit_basis([episodes[0],episodes[1]*2]);np.testing.assert_allclose(basis.mean,self.basis.mean,atol=1e-14);np.testing.assert_allclose(basis.rms,self.basis.rms,atol=1e-14);self.assertEqual(report['floor_fraction'],.1);self.assertGreaterEqual(basis.rms.min(),report['feature_RMS_floor']);np.testing.assert_allclose(self.F,np.array([self.basis.features(s) for e in episodes for s in e]),atol=1e-14);raw=self.F.T@(self.weights[:,None]*self.F);np.testing.assert_allclose(self.G,raw/(np.trace(raw)/256)+.01*np.eye(256),atol=1e-14)

    def test_zero_literal_ties_nonzero_serialization_raw_reuse_rejected(self):
        tiny=FixedPairwise(np.array([[0,1],[1,0]]),np.ones((2,2)),np.ones(2),np.zeros(2),np.ones(4));original=FrozenPairwiseReadout(np.zeros((3,2)),tiny,np.ones((3,4))*.3);zero=ContinuousReadout(original,self.basis,np.zeros((3,256)));encoded=dict(indices=torch.tensor([0]),values=torch.tensor([1.]),entities=[]);a,q=original.choose(np.array([0]),np.array([1.]),[2,0]);got,zq=zero.choose(np.array([0]),np.array([1.]),[2,0]);self.assertEqual(a,got);np.testing.assert_array_equal(q,zq);self.assertEqual(zero.choose_encoded(encoded,None)[0],0)
        nonzero=ContinuousReadout(original,self.basis,np.ones((3,256)));self.assertTrue(nonzero.requires_structured)
        with self.assertRaises(ValueError):nonzero.choose(np.array([0]),np.array([1.]))
        cert=certify_episode({},None,None,None,source_weights_sha256='x',source_fingerprint='x',candidate_fingerprint='x',candidate_config={},candidate_readout=nonzero);self.assertFalse(cert['certified']);self.assertIn('structured',cert['reason'])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.npz';save_readout(p,nonzero);restored=load_readout(p);np.testing.assert_array_equal(restored.output,nonzero.output);self.assertEqual(restored.schema,nonzero.schema);scene=calibration()[0][0];np.testing.assert_array_equal(restored.continuous.features(scene),nonzero.continuous.features(scene));p=Path(d)/'scenes.npz';save_scenes(p,[scene,np.empty((0,6))]);scenes=load_scenes(p);np.testing.assert_array_equal(scenes[0],scene);self.assertEqual(scenes[1].shape,(0,6))

    def test_output_analytic_score(self):
        f=self.F[0];rng=np.random.default_rng(8);C=rng.normal(size=(2,256))*.01;base=np.array([.2,-.1,.3]);q=base+CONTRAST@C@f;T=.8;a=1;analytic=score(a,probabilities(q,T),f,T);eps=1e-6
        for i,j in [(0,0),(1,173),(0,255)]:
            plus=C.copy();minus=C.copy();plus[i,j]+=eps;minus[i,j]-=eps;numeric=(np.log(probabilities(base+CONTRAST@plus@f,T)[a])-np.log(probabilities(base+CONTRAST@minus@f,T)[a]))/(2*eps);self.assertAlmostEqual(analytic[i,j],numeric,places=8)


if __name__=='__main__':unittest.main()
