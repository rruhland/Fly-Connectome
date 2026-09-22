import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from test_short_term import make, inject
from short_term import FacilitationNetwork
from signed_kinetics import AreaMatchedKineticsNetwork
from frame_prediction import FramePrediction
from fly_connectome.plasticity import LearningConfig
from credit_replay import CreditRecorder, challenge_frames


@pytest.mark.parametrize('cls', [AreaMatchedKineticsNetwork, FacilitationNetwork])
def test_credit_recorder_preserves_exact_learning_and_tags_issue_state(cls):
    nets = [make(cls), make(cls)]
    cfg = LearningConfig(prediction_encoding='signed-current-v1',
        visual_target='input-arrivals-v1', visual_eligibility='forecast-causal-v1',
        eta_prediction=.1, homeostasis_rate=0.)
    rules = [FramePrediction(nets[0], cfg), CreditRecorder(nets[1], cfg, audit_edges=[0, 1])]
    for tick in range(40):
        for net, rule in zip(nets, rules):
            if tick in [0, 8, 16, 24]:
                inject(net, tick)
            net.sensory_state[0, 2] = tick  # overwritten by each normal step
            a = net.step(torch.tensor([[0., 0., float(tick)]]), capture_increments=True)
            rule.observe(a, torch.zeros(1))
            if tick % 8 == 7:
                rule.synchronize()
        for name in ['magnitudes', 'voltage', 'predictive_current']:
            torch.testing.assert_close(getattr(nets[0], name), getattr(nets[1], name), rtol=0, atol=0)
        torch.testing.assert_close(rules[0].proposals, rules[1].proposals, rtol=0, atol=0)
    rows = rules[1].rows
    assert [r['issue_tick'] for r in rows] == [0, 8, 16, 24]
    assert [r['confirm_tick'] for r in rows] == [8, 16, 24, 32]
    assert rows[0]['issue_state'][0, 0] == 0
    assert rows[0]['confirm_state'][0, 0] == 8
    assert all(r['accumulator_error'] == 0 for r in rows)


def test_challenges_preserve_binary_camera_and_repeat_column_polarity():
    a, info = challenge_frames('repeat', [2], [2], (30, 39), (30, 40))
    assert a.dtype == torch.bool
    occupancy = a[:, 0].sum((1, 2)).numpy()
    changes = np.diff(np.r_[0, occupancy])
    np.testing.assert_array_equal(changes[changes != 0], np.tile([1, 1, -1, -1], 2))
    standard, _ = challenge_frames('standard', [2], [2], (30, 39), (30, 40))
    omitted, _ = challenge_frames('omitted', [2], [2], (30, 39), (30, 40))
    assert len(a) == len(standard) == len(omitted) == len(info['trial'])
    assert (standard != omitted).any()
