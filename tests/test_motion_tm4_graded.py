import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from motion_tm4_graded import future_deviation_events, future_events_at, roc_auc


def test_roc_auc_counts_tied_scores_as_half_a_pair():
    scores = np.array([0., 1., 1., 2.])
    events = np.array([False, False, True, True])
    assert roc_auc(scores, events) == .875


def test_future_deviation_events_subtracts_matching_blank_before_window_max():
    import torch

    stimulus = torch.zeros(144, 1)
    blank = torch.zeros_like(stimulus)
    stimulus[32, 0] = .03
    blank[32, 0] = .02
    stimulus[48, 0] = .03
    blank[48, 0] = .01
    events = future_deviation_events(stimulus, blank, threshold=.015)
    assert not bool(events[0, 0])
    assert not bool(events[1, 0])
    assert bool(events[2, 0])


def test_future_events_at_uses_fixed_eight_tick_window_after_horizon():
    import torch

    spikes = torch.zeros(144, 1)
    spikes[56, 0] = 1
    delayed = future_events_at(spikes, horizon=32)
    near = future_events_at(spikes, horizon=8)
    assert bool(delayed[0, 0])
    assert not bool(delayed[1, 0])
    assert bool(near[3, 0])
