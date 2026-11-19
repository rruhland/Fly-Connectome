import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))
from pong_motor_recalibration import BoundedCategoricalMotor, MotorGainLearner


def test_forgetting_preserves_bounded_uncertainty_in_unexcited_directions():
    m = BoundedCategoricalMotor(5)
    x = m.interval_features([1]*32,.02)
    for _ in range(1280):
        m.observe(x,.012,.02)
    values = np.linalg.eigvalsh(m.covariance)
    assert values.min() > 0
    assert values.max() <= m.settings['initial_covariance'] + 1e-8
    assert abs(m.predict(x)-.012) < .001


def test_recovery_preserves_acquired_shape_and_resumes_mid_probe():
    m = MotorGainLearner(6)
    m.theta = np.concatenate([np.linspace(0,.01,32),np.zeros(32),np.linspace(0,-.01,32)])
    m.updates = 600
    m.mismatch_count = 5
    m.feedback(1, -1, True, False)
    assert not m.kernel
    m.mismatch_count = 5
    m.feedback(0, -1, True, False)
    assert m.calibrations == 1 and len(m.probe_sequence) == 36
    other = MotorGainLearner.from_state(json.loads(json.dumps(m.state())))
    frozen = m.frozen_branch()
    for i in range(48):
        assert m.perturb(1,True) == other.perturb(1,True)
        x = m.interval_features([i%3-1]*32,.02)
        assert m.observe(x,.021,.02) == other.observe(x,.021,.02)
        frozen.observe(x,.021,.02)
    assert m.state() == other.state()
    assert frozen.updates == 600
    assert len(m.gains)==3 and np.allclose(np.sum(m.kernel),1.)
