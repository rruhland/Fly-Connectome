import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from motion_graded_propagation import (causal_highpass, dominant_target_masks,
                                       select_cap, summarize_local_candidate,
                                       summarize_target)


def test_blank_only_calibration_selects_largest_quiet_cap():
    rows = [dict(cap=.01, blank_p99_impulse=.02, blank_spike_rate=0., finite=True),
            dict(cap=.03, blank_p99_impulse=.04, blank_spike_rate=.005, finite=True),
            dict(cap=.1, blank_p99_impulse=.07, blank_spike_rate=.002, finite=True)]
    assert select_cap(rows) == .03
    assert select_cap(rows[2:]) is None


def test_target_summary_preserves_signed_current_and_per_cell_response():
    blank = dict(impulse=torch.zeros(4, 3), current=torch.zeros(4, 3),
                 voltage=torch.zeros(4, 3),
                 spikes=torch.zeros(4, 3, dtype=torch.bool))
    stimulus = {name: value.clone() for name, value in blank.items()}
    stimulus['impulse'][:, 0] = .2
    stimulus['impulse'][:, 1] = -.1
    stimulus['current'][:, :2] = .3
    stimulus['voltage'][:, :2] = .02
    stimulus['spikes'][1, 0] = True
    row = summarize_target(stimulus, blank, torch.tensor([0, 1]),
                           start=0, end=4)
    assert abs(row['impulse']['mean_signed_change']-.05) < 1e-7
    assert abs(row['impulse']['mean_absolute_change']-.15) < 1e-7
    assert abs(row['voltage']['per_cell_absolute_p90']-.02) < 1e-7
    assert row['stimulus_spikes'] == 1
    assert row['blank_spikes'] == 0


def test_highpass_uses_only_past_local_state():
    trace = torch.tensor([[1.], [0.], [1.]])
    positive, negative = causal_highpass(trace, torch.tensor([0.]), decay=.5)
    assert torch.equal(positive[:, 0], torch.tensor([1., 0., .75]))
    assert torch.equal(negative[:, 0], torch.tensor([0., .5, 0.]))


def test_local_candidate_screen_uses_own_blank_floor():
    blank = torch.tensor([[0., 0.], [1., 0.], [0., 0.], [1., 0.]])
    stimulus = blank + torch.tensor([2., 1.])
    row = summarize_local_candidate(stimulus, blank,
                                    torch.tensor([0, 1]), start=0, end=4)
    assert row['fraction_above_blank_p99'] == 1.
    assert row['blank_fraction_above_blank_p99'] == 0.
    assert row['mean_absolute_change'] == 1.5


def test_local_target_groups_come_only_from_measured_contact_mass():
    pre = np.array([0, 0, 1, 1])
    post = np.array([2, 3, 2, 3])
    contacts = np.array([5, 1, 2, 7])
    groups, masses = dominant_target_masks(pre, post, contacts,
        {18: np.array([True, False, False, False]),
         46: np.array([False, True, False, False])},
        np.array([False, False, True, True]))
    assert groups[18].tolist() == [2]
    assert groups[46].tolist() == [3]
    assert masses[18].tolist() == [5, 1]
    assert masses[46].tolist() == [2, 7]
