import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from motion_lc11_inhibitory_voltage import (CLASSES, inhibitory_candidate,
                                            summarize_class)


def test_summary_weights_only_measured_lc11_source_contacts():
    blank = dict(voltage=torch.zeros(144, 3),
                 spikes=torch.zeros(144, 3, dtype=torch.bool), pixel_events=0)
    dot = dict(voltage=blank['voltage'].clone(),
               spikes=blank['spikes'].clone(), pixel_events=72)
    dot['voltage'][24:120, 0] = .003
    dot['voltage'][24:120, 1] = -.001
    result = summarize_class(dot, blank, torch.tensor([True, True, False]),
                             np.array([3., 1., 100.]))
    assert result['neurons'] == 2
    assert result['contacts'] == 4
    assert abs(result['weighted_signed_voltage']-.002) < 1e-7
    assert result['cells_above_002'] == 1
    assert result['positive_cells_above_002'] == 1


def test_candidate_requires_bar_specific_signal_at_both_locations():
    rows = {}
    for center in (18, 46):
        for polarity in ('on', 'off'):
            for kind in ('dot', 'bar'):
                rows[f'{center}-{polarity}-{kind}'] = {
                    label: dict(positive_cells_above_002=(5 if label == 'Li15'
                                                         and kind == 'bar' else 0),
                                weighted_signed_voltage=(.004 if label == 'Li15'
                                                          and kind == 'bar' else .001))
                    for label in CLASSES}
    assert inhibitory_candidate(rows) == dict(cell_type='Li15', polarity='on')
    rows['46-on-bar']['Li15']['weighted_signed_voltage'] = .002
    assert inhibitory_candidate(rows) == dict(cell_type='Li15', polarity='off')
