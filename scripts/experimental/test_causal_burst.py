import tempfile,time,unittest
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from pong_fixed_pairwise import FixedPairwise,FrozenPairwiseReadout
from pong_joint_output import fit
from pong_causal_burst import improved,audited_cases
import run_practical_pong_v2 as base


class CausalBurstTests(unittest.TestCase):
    def policies(self):
        b=FixedPairwise(np.array([[0,1],[1,0]]),np.ones((2,2)),np.ones(2),np.zeros(2),np.ones(3));W=np.zeros((3,2));W[1,0]=1.;D=np.zeros((3,2));D[0,0]=2.;return FrozenPairwiseReadout(W,b,np.zeros((3,3))),FrozenPairwiseReadout(D,b,np.zeros((3,3)))

    def test_credit_is_whole_horizon_increment_not_any_previous_hit(self):
        self.assertFalse(improved(dict(complete=True,hits=5),4,2));self.assertTrue(improved(dict(complete=True,hits=7),4,2));self.assertFalse(improved(dict(complete=False,hits=8),4,2))

    def test_fit_rejects_wrong_greedy_without_honest_reward_provenance(self):
        p,d=self.policies();c=dict(seed=1,X=sp.csr_matrix([[1.,0.]]),actions=np.array([0]),source_readout=p,source_policy_sha256='current',interventions=[])
        with self.assertRaises(AssertionError):fit(p,[c],np.ones(2),1.,np.eye(5),.1,time.monotonic()+5,'current')
        c['causal_audit_passed']=True;c['interventions']=[dict(row=0,action=0,source_action=1,reward_verified=False,native_total_hit_advantage=1)]
        with self.assertRaises(AssertionError):fit(p,[c],np.ones(2),1.,np.eye(5),.1,time.monotonic()+5,'current')
        c['interventions'][0]['reward_verified']=True;result,r=fit(p,[c],np.ones(2),1.,np.eye(5),.1,time.monotonic()+5,'current');self.assertTrue(r['verified']);self.assertEqual(r['source_exact_choose_rows'],0);self.assertEqual(len(r['reward_verified_intervention_rows']),1);new=FrozenPairwiseReadout(result[0],p.basis,result[1]);a,q=new.choose(np.array([0]),np.array([1.]));self.assertEqual(a,0);self.assertGreaterEqual(q[0]-max(q[1:]),.1-1e-9)

    def test_intervention_audit_checks_source_Q_and_donor_choice(self):
        p,d=self.policies()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);X=sp.csr_matrix([[1.,0.]]*3);sp.save_npz(root/'raw.npz',X);source=[dict(action=1,q=[0.,1.,0.],hits=int(i==2)) for i in range(3)];records=[dict(v) for v in source];records[1].update(action=0,hits=1,intervention_action=0,unmodified_greedy_action=1,donor_policy_action=0);base.atomic_json(root/'source.json',dict(weights_sha256='current',records=source,report=dict(hits=1,config={})));base.atomic_json(root/'records.json',records)
            r=dict(seed=1,complete=True,positive=True,total_hit_advantage=1,same_original_episode_horizon=True,source_policy_sha256='current',donor_policy_sha256='donor',raw_feature=str(root/'raw.npz'),records=str(root/'records.json'),original_receipt=str(root/'source.json'),original_raw_feature=str(root/'raw.npz'),report=dict(complete=True,hits=2,config={}),branch_total_hits=2,transition=1,mask=[True]+[False]*9,remaining_changed_choices=1)
            cases,audit=audited_cases(r,p,d,'current','donor');self.assertEqual(audit['rollin_rows'],1);self.assertEqual(audit['requested_changed_rows'],[1]);self.assertEqual(len(cases[-1]['interventions']),1)
            records[1]['q']=[0.,2.,0.];base.atomic_json(root/'records.json',records)
            with self.assertRaises(AssertionError):audited_cases(r,p,d,'current','donor')


if __name__=='__main__':unittest.main()
