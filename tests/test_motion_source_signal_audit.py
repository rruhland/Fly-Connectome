import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from motion_source_signal_audit import summarize_feature


def test_summary_compares_matched_ticks_and_local_noise_floor():
    blank = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                          [0., 0., 0.], [1., 0., 0.]])
    stimulus = blank.clone()
    stimulus[:, 0] += 2.
    stimulus[:, 1] += 1.
    stimulus[:, 2] += .25
    row = summarize_feature(stimulus, blank, torch.tensor([0, 1]),
                            torch.tensor([2]), start=0, end=4)
    assert row['local_mean_absolute_change'] == 1.5
    assert row['local_mean_signed_change'] == 1.5
    assert row['remote_mean_absolute_change'] == .25
    assert row['local_fraction_above_blank_p99'] == 1.
    assert row['blank_fraction_above_blank_p99'] == 0.
    assert abs(row['per_cell_signed_change_p10']-1.1) < 1e-6
    assert row['per_cell_signed_change_p50'] == 1.5
    assert abs(row['per_cell_signed_change_p90']-1.9) < 1e-6
    assert row['finite']
