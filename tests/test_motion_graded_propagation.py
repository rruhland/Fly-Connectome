import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from motion_graded_propagation import select_cap, summarize_target


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
