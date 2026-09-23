import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from graded_mi4 import GradedMi4Network, blank_limited_release
from motion_stage_graded import select_fraction
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph


def test_graded_release_is_nonnegative_delayed_and_inhibitory():
    graph = Graph(np.array([1, 2, 3]), np.array([0, 1]), np.array([0, 2]),
                  np.array([1, 1]), np.array([1, -1, 1]), 1.)
    net = GradedMi4Network(graph, [1, 1], ['predictive', 'feedforward'],
        config=NeuronConfig(tau_sensory=.02), cell_types=['L1', 'Mi4', 'T4a'],
        target_mask=torch.tensor([True, False, False]), base_release=.1,
        voltage_scale=.1, release_fraction=1.)
    assert 0 < blank_limited_release(net) < .1
    zero = torch.zeros(1, 3)
    net.enable_graded()
    net.step(zero)
    assert torch.isclose(net.last_graded_impulse[0, 2], torch.tensor(-.1))
    net.voltage[0, 1] = -.5
    net.step(zero)
    assert net.graded_history[1, 0] == 0
    net.step(zero)
    assert net.last_graded_impulse[0, 2] == 0
    assert torch.all(net.graded_history >= 0)


def test_flash_only_selection_respects_blank_and_both_locations():
    rows = [dict(release_fraction=value, blank_p99_current=blank,
                 flash_modulation={'18': left, '46': right}, finite=True)
            for value, blank, left, right in ((.25, .04, .007, .003),
                                               (.5, .06, .01, .01),
                                               (1., .045, .008, .006))]
    assert select_fraction(rows) == 1.
    assert select_fraction(rows[:2]) is None
