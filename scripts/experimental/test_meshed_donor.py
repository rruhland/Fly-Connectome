import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from run_meshed_donor_pong import refilter


def test_repair_eligibility_and_action_targets_follow_current_carry():
    theta=np.array([[1.,0.],[0.,1.],[-1.,-1.]])
    old=dict(phase='left',seed=7,X=np.eye(2),actions=np.array([0,0]))
    current=refilter(old,theta,{'7':0})
    assert current['X'].shape==(1,2) and np.array_equal(current['X'][0],[0.,1.]) and current['actions'][0]==0
    assert refilter(old,theta,{'7':1}) is None
    aligned=dict(old,actions=np.array([0,1]));assert refilter(aligned,theta,{'7':0}) is None
