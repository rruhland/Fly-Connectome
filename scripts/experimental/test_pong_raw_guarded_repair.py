import time,unittest,tempfile,json
from pathlib import Path
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

    def test_early_feasible_repair_preserves_guard_actions(self):
        W=np.array([[1.,0.],[0.,0.],[-1.,0.]])
        g=[dict(seed=1,group='first_left',X=sp.csr_matrix([[1.,0.]]),actions=np.array([0]))]
        p=[dict(seed=2,group='later_left',X=sp.csr_matrix([[0.,1.]]),actions=np.array([2]))]
        d,r=fit(W,p,g,dict(input_rms=1.,kappa=.03),time.monotonic()+3,stop_on_earned_repair=True)
        self.assertEqual(r['status'],'feasible_earned_repair_early');self.assertEqual(r['guard_action_changes'],0)
        self.assertEqual(r['positive']['requested_action_fraction'],1.);self.assertLessEqual(r['final_objective'],r['zero_residual_objective'])

    def test_smaller_margin_still_preserves_exact_choices(self):
        W=np.array([[1.,0.],[0.,0.],[-1.,0.]])
        g=[dict(seed=1,group='first_left',X=sp.csr_matrix([[1.,0.]]),actions=np.array([0]))]
        p=[dict(seed=2,group='later_left',X=sp.csr_matrix([[0.,1.]]),actions=np.array([2]))]
        d,r=fit(W,p,g,dict(input_rms=1.,kappa=.03),time.monotonic()+3,stop_on_earned_repair=True,guard_margin_scale=1e-4)
        self.assertEqual(r['guard_action_changes'],0);self.assertEqual(r['positive']['requested_action_fraction'],1.)
        self.assertAlmostEqual(r['required_margin_max'],3e-6)

    def test_exact_conflicting_rows_rejected_without_nearby_matching(self):
        from rally_witness_replay import conflicts
        p=[dict(X=sp.csr_matrix([[1.,0.]]),actions=np.array([0]))]
        same=[dict(X=sp.csr_matrix([[1.,0.]]),actions=np.array([1]))]
        nearby=[dict(X=sp.csr_matrix([[1.+1e-12,0.]]),actions=np.array([1]))]
        self.assertFalse(conflicts(p,same)['compatible']);self.assertTrue(conflicts(p,nearby)['compatible'])

    def test_target_guard_stops_at_last_actual_return(self):
        from run_rally_guarded_curriculum import earned_target_witness
        with tempfile.TemporaryDirectory() as path:
            folder=Path(path);sp.save_npz(folder/'raw.npz',sp.eye(7,format='csr'))
            receipt=dict(seed=42,feature_file='raw.npz',report=dict(hits=2),records=[dict(action=i%3,hits=int(i in (1,4))) for i in range(7)])
            self.assertIsNone(earned_target_witness(folder,receipt,2,'later','right'))
            witness=earned_target_witness(folder,receipt,1,'later','right')
            self.assertEqual(witness['hit_indices'],[1,4]);self.assertEqual(witness['target_hits'],2)
            self.assertEqual(sp.load_npz(folder/witness['path']).shape[0],5)
            np.testing.assert_array_equal(np.load(folder/witness['actions_path']),[0,1,2,0,1])

    def test_saved_witness_skips_already_earned_count(self):
        from rally_witness_replay import select_saved
        with tempfile.TemporaryDirectory() as path:
            folder=Path(path);(folder/'manifest.json').write_text(json.dumps(dict(source={})))
            (folder/'summary.json').write_text(json.dumps(dict(source_frozen=True,witnesses=[dict(seed=42,path='unused',target_hits=2)])))
            chosen,screen=select_saved([folder],np.zeros((3,2)),[],[],set(),{},known_hits={42:2})
            self.assertIsNone(chosen);self.assertIn('already earned',screen[0]['reason'])

    def test_vector_metric_changes_training_coordinates_not_policy_inputs(self):
        W=np.array([[1.,0.],[0.,0.],[-1.,0.]])
        g=[dict(seed=1,group='first_left',X=sp.csr_matrix([[1.,0.]]),actions=np.array([0]))]
        p=[dict(seed=2,group='later_right',X=sp.csr_matrix([[0.,1.]]),actions=np.array([2]))]
        d,r=fit(W,p,g,dict(input_rms=1.,kappa=.03),time.monotonic()+3,stop_on_earned_repair=True,guard_margin_scale=1e-4,feature_scale=np.array([3.,.2]))
        self.assertEqual(r['guard_action_changes'],0);self.assertEqual(r['positive']['requested_action_fraction'],1.)
        np.testing.assert_allclose(d.sum(0),0,atol=1e-15);self.assertAlmostEqual(r['feature_scale_max'],3.)

if __name__=='__main__':unittest.main()
