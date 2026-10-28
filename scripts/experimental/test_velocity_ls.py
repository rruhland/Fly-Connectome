import unittest
import torch
from fly_connectome.vision.interface import VisualStateEncoder
from pong_velocity_ls import FourObservationEncoder,synthetic_diagnostic


class VelocityLSTest(unittest.TestCase):
    def test_constant_motion_reversal_and_reset_are_measured(self):
        r=synthetic_diagnostic();self.assertTrue(r['passed']);self.assertEqual(r['sign_reversal_lag_samples'],1)

    def test_only_existing_velocity_slots_change(self):
        old=VisualStateEncoder(64,64,.02);new=FourObservationEncoder(64,64,.02)
        for sample,x in enumerate((25.,25.5,26.,26.5,26.)):
            state=dict(sample=sample,forecasts=[],entities=[dict(id=1,position=torch.tensor([30.,x]),observed=True,observation_age=0)]);a=old.encode(state,None);b=new.encode(state,None);av=torch.zeros(4389);bv=torch.zeros(4389);av[a['indices']]=a['values'];bv[b['indices']]=b['values'];mask=torch.ones(4389,dtype=torch.bool);lo,hi=old.layout['current'];mask[lo+2*256:hi]=False;lo,hi=old.layout['relations'];mask[lo+81:hi]=False;self.assertTrue(torch.equal(av[mask],bv[mask]));self.assertEqual(a['dimension'],b['dimension'])


if __name__=='__main__':unittest.main()
