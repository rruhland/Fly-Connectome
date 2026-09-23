import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from ct1_column_shadow import CT1ColumnShadow


def test_ct1_shadow_requires_two_causal_legs_and_stays_in_its_column():
    network = SimpleNamespace(n=3, history_length=3, step_index=0,
        config=SimpleNamespace(dt=.001, tau_current=.005),
        history=torch.zeros((3, 1, 3), dtype=torch.bool),
        release_history=torch.zeros((3, 0)),
        current_decay=torch.full((3,), .5),
        feedforward_current=torch.zeros((1, 3)))
    shadow = CT1ColumnShadow(network, 2,
        tm1_nodes=[0], tm1_columns=[0], tm1_weights=[1.],
        tm9_indices=[], tm9_columns=[], tm9_weights=[],
        output_posts=[1, 2], output_columns=[0, 1], output_weights=[2., 2.])
    shadow.enabled = True
    shadow.begin_tick(network, output_enabled=True)
    assert torch.count_nonzero(shadow.last_impulse) == 0
    network.history[0, 0, 0] = True
    network.step_index = 1
    shadow.begin_tick(network, output_enabled=True)
    assert torch.count_nonzero(shadow.last_impulse) == 0
    network.step_index = 2
    shadow.begin_tick(network, output_enabled=True)
    assert torch.isclose(shadow.last_impulse[1], torch.tensor(-.2))
    assert shadow.last_impulse[2] == 0
    assert torch.isclose(network.feedforward_current[0, 1], torch.tensor(-.4))


def test_ct1_shadow_off_arm_processes_input_without_injecting_current():
    network = SimpleNamespace(n=2, history_length=2, step_index=1,
        config=SimpleNamespace(dt=.001, tau_current=.005),
        history=torch.tensor([[[True, False]], [[False, False]]]),
        release_history=torch.zeros((2, 0)),
        current_decay=torch.ones(2),
        feedforward_current=torch.zeros((1, 2)))
    shadow = CT1ColumnShadow(network, 1,
        tm1_nodes=[0], tm1_columns=[0], tm1_weights=[1.],
        tm9_indices=[], tm9_columns=[], tm9_weights=[],
        output_posts=[1], output_columns=[0], output_weights=[1.])
    shadow.enabled = True
    shadow.begin_tick(network, output_enabled=False)
    assert shadow.last_release[0] > 0
    assert network.feedforward_current[0, 1] == 0
