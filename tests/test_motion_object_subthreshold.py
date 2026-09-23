import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from motion_object_subthreshold import passes_object_screen, summarize_signal


def test_subthreshold_summary_uses_matched_tickwise_blank_and_remote_cells():
    blank = dict(voltage=torch.zeros(144, 4), current=torch.zeros(144, 4),
                 spikes=torch.zeros(144, 4, dtype=torch.bool), pixel_events=0)
    stimulus = dict(voltage=blank['voltage'].clone(),
                    current=blank['current'].clone(),
                    spikes=blank['spikes'].clone(), pixel_events=72)
    stimulus['voltage'][24:120, :2] = .003
    stimulus['voltage'][24:120, 2:] = .001
    stimulus['current'][24:120, :2] = .1
    result = summarize_signal(stimulus, blank, torch.tensor([0, 1]),
                              torch.tensor([2, 3]))
    assert abs(result['voltage']['local_mean_absolute']-.003) < 1e-7
    assert abs(result['voltage']['remote_mean_absolute']-.001) < 1e-7
    assert result['voltage']['local_cells_above_002'] == 2
    assert result['voltage']['peak_tick'] == 24
    assert result['pixel_events'] == 72


def test_screen_requires_both_polarities_at_both_locations():
    rows = {}
    for center in (18, 46):
        for polarity in ('on', 'off'):
            rows[f'{center}-{polarity}-dot'] = {
                label: dict(voltage=dict(local_mean_absolute=(.003 if label == 'T2'
                                                              else 0.),
                                         remote_mean_absolute=.001,
                                         local_cells_above_002=(5 if label == 'T2'
                                                                else 0)))
                for label in ('T2', 'T2a', 'T3')}
    assert passes_object_screen(rows)
    rows['46-off-dot']['T2']['voltage']['local_cells_above_002'] = 4
    assert not passes_object_screen(rows)
