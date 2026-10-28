import unittest
import torch
import json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
from fly_connectome.vision.interface import VisualStateEncoder
from pong_velocity_ls import FourObservationEncoder,synthetic_diagnostic


class VelocityLSTest(unittest.TestCase):
    def test_constant_motion_reversal_and_reset_are_measured(self):
        r=synthetic_diagnostic();self.assertTrue(r['passed']);self.assertEqual(r['sign_reversal_lag_samples'],1)

    def test_only_existing_velocity_slots_change(self):
        old=VisualStateEncoder(64,64,.02);new=FourObservationEncoder(64,64,.02)
        for sample,x in enumerate((25.,25.5,26.,26.5,26.)):
            state=dict(sample=sample,forecasts=[],entities=[dict(id=1,position=torch.tensor([30.,x]),observed=True,observation_age=0)]);a=old.encode(state,None);b=new.encode(state,None);av=torch.zeros(4389);bv=torch.zeros(4389);av[a['indices']]=a['values'];bv[b['indices']]=b['values'];mask=torch.ones(4389,dtype=torch.bool);lo,hi=old.layout['current'];mask[lo+2*256:hi]=False;lo,hi=old.layout['relations'];mask[lo+81:hi]=False;self.assertTrue(torch.equal(av[mask],bv[mask]));self.assertEqual(a['dimension'],b['dimension'])

    def test_nondefault_direct_cache_cannot_override_default_source_episode(self):
        from run_rally_guarded_curriculum import evaluate
        import run_practical_pong_v2 as base
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);head=root/'head.npy';np.save(head,np.zeros((3,1)));feature=root/'raw.npz';sp.save_npz(feature,sp.csr_matrix([[1.]]));sha=base.sha(head);config=dict(policy_kind='trained',action_hold_frames=1,episodes=1,action_seed=None);receipt=dict(seed=1,complete=True,config=config,weights_sha256=sha,feature_format='actual capped raw CSR',transitions=1,initial_cameras=1,records=[dict(action=0,q=[0.,0.,0.])],report=dict(complete=True),feature_file=str(feature));cached=root/'cached';(cached/'native_candidate/episodes').mkdir(parents=True);base.atomic_json(cached/'manifest.json',dict(source=base.source_signature()));base.atomic_json(cached/'native_candidate/episodes/train_00.json',dict(receipt,transport_kind='visual-transport-v1:observed-four-point-LS-velocity'));entry=dict(seed=1,receipt='baseline',raw_feature=str(feature));bank=dict(current_head=str(head),current_head_sha256=sha,source=base.source_signature())
            with patch('run_rally_guarded_curriculum.ProcessPoolExecutor') as pool:
                outcomes,cost=evaluate(root/'result',bank,[entry],[receipt],head,cached_native_root=cached)
                pool.return_value.__enter__.return_value.submit.assert_not_called()
            self.assertEqual(outcomes[0]['execution'],'reused_by_exact_action_certificate');self.assertEqual(cost['reused_completed_native_games'],0)


if __name__=='__main__':unittest.main()
