import tempfile,unittest
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from pong_fixed_pairwise import FixedPairwise,save_readout
from run_rally_joint_block import receipt_entry,next_prefix,memory_counts,remember,violations
import run_practical_pong_v2 as base


class JointBlockMemoryTests(unittest.TestCase):
    def fixture(self,root,name,hits):
        folder=root/name;folder.mkdir();W=np.zeros((3,2));W[1]=len(name);np.save(folder/'proposal.npy',W);basis=FixedPairwise(np.array([[0,1],[1,0]]),np.ones((2,2)),np.ones(2),np.zeros(2),np.ones(3));R=np.full((3,3),len(name)/10);save_readout(folder/'model.npz',W,basis,R);records=[dict(hits=int(i%2==1)) for i in range(2*hits)];sp.save_npz(folder/'raw.npz',sp.csr_matrix((2*hits,2)));base.atomic_json(folder/'receipt.json',dict(seed=1,complete=True,weights_sha256=base.sha(folder/'model.npz'),base_weights_sha256=base.sha(folder/'proposal.npy'),feature_file='raw.npz',records=records,report=dict(hits=hits)))
        return receipt_entry(folder/'receipt.json',folder,folder/'proposal.npy',folder/'model.npz')

    def test_actual_new_target_baseline_can_make_source_redundant(self):
        with tempfile.TemporaryDirectory() as d:
            source=self.fixture(Path(d),'source',3)
            self.assertIsNone(next_prefix(source,4));self.assertIsNone(next_prefix(source,3))
            next_case=next_prefix(source,1);self.assertEqual(next_case['end'],4);self.assertEqual(next_case['earned_returns'],2);self.assertEqual(source['earned_returns'],3)

    def test_mixed_source_identity_and_only_earned_improvements_are_retained(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=self.fixture(root,'old',1);b=self.fixture(root,'newer',3);memory={1:a};remember(memory,b)
            self.assertEqual(memory[1]['source_model_bundle'],b['source_model_bundle']);self.assertNotEqual(a['source_head_sha256'],b['source_head_sha256']);self.assertEqual(a['earned_returns'],1)
            remember(memory,a);self.assertEqual(memory[1]['source_policy_sha256'],b['source_policy_sha256']);counts=memory_counts(memory);self.assertEqual(counts['returns'],3);self.assertTrue(counts['guaranteed_lower_bounds_not_current_full_native_outcomes']);self.assertNotIn('later_opportunities',counts)

    def test_per_seed_loss_is_detected_even_if_aggregate_returns_increase(self):
        memory={1:dict(earned_returns=2),2:dict(earned_returns=3)};rows=[dict(seed=1,complete=True,report=dict(hits=6)),dict(seed=2,complete=True,report=dict(hits=2))];initial={1:1,2:1};self.assertTrue(all(r['report']['hits']>=initial[r['seed']] for r in rows));self.assertGreater(sum(r['report']['hits'] for r in rows),5);self.assertEqual(violations(rows,memory),[dict(seed=2,minimum=3,actual=2)])


if __name__=='__main__':unittest.main()
