import sys
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('scipy')
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from context_audit import signed_prediction, threshold_for_tempos
from local_information import NeighborProbe


def test_signed_probe_separates_polarity_from_event_threshold():
    x=np.array([[-2.],[-1.9],[0.],[.1],[2.],[2.1]])
    y=np.array([-1,-1,0,0,1,1])
    probe=NeighborProbe(x,y,k=1)
    scores,pred=signed_prediction(probe,np.array([[-2.],[0.],[2.]]),y,.5)
    np.testing.assert_array_equal(scores,[1,0,1])
    np.testing.assert_array_equal(pred,[-1,0,1])


def test_calibration_cap_is_enforced_separately_for_each_tempo():
    tempo=np.repeat([2,4,6],22)
    y=np.tile(np.r_[-1,1,np.zeros(20)],3)
    scores=np.tile(np.r_[.9,.9,np.zeros(20)],3)
    scores[2:4]=.95  # Two false positives in one tempo; pooled rate would hide them.
    threshold=threshold_for_tempos(scores,y,tempo)
    assert threshold>.95
