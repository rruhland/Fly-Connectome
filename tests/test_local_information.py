import sys
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('scipy')  # Diagnostic uses the existing optional data extra.

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from local_information import NeighborProbe, calibrate_threshold, detection_metrics


def test_probe_uses_training_scaling_and_handles_constant_features():
    x=np.array([[0.,4.],[.1,4.],[.2,4.],[2.,4.],[2.1,4.],[2.2,4.]])
    y=np.array([0,0,0,1,1,1])
    probe=NeighborProbe(x,y,k=3)
    np.testing.assert_array_equal(probe.score(np.array([[.15,999.],[2.15,-999.]])),[0.,1.])
    before=probe.scale.copy()
    probe.score(np.array([[100000.,4.]]))
    np.testing.assert_array_equal(before,probe.scale)


def test_calibration_retains_both_polarities_and_counts_quiet_frames():
    labels=np.array([-1,-1,1,1,0,0,0,0])
    scores=np.array([.9,.8,.7,.6,.5,.4,.3,.2])
    threshold=calibrate_threshold(scores,labels)
    metrics=detection_metrics(scores,labels,threshold)
    assert metrics['on_recall']==metrics['off_recall']==1
    assert metrics['quiet_false_positive_rate']==0
    assert metrics['quiet_count']==4
    # Lowering threshold does not hide quiet errors in aggregate accuracy.
    assert detection_metrics(scores,labels,.2)['quiet_false_positive_rate']==1
