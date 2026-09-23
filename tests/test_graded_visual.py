import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from graded_visual import GradedVisualNetwork, graded_release
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph


def test_local_voltage_release_has_a_noise_floor_and_bounded_nonnegative_output():
    value = graded_release(torch.tensor([.02, .004, -.01]),
                           torch.zeros(3), torch.tensor([.005, .005, .005]),
                           scale=.01, cap=.1)
    assert torch.equal(value, torch.tensor([.1, 0., 0.]))


@pytest.mark.parametrize('transmitter_sign', [-1, 1])
def test_existing_t2_edges_keep_pathway_sign_delay_and_weight(transmitter_sign):
    graph = Graph(np.arange(1, 6), np.array([0, 1, 1, 1]),
                  np.array([4, 2, 3, 4]), np.ones(4, dtype=np.int64),
                  np.array([1, transmitter_sign, 1, 1, 1]), 1.)
    net = GradedVisualNetwork(graph, [1, 1, 2, 1],
        ['predictive', 'feedforward', 'predictive', 'predictive'],
        config=NeuronConfig(tau_sensory=.02, threshold=100.),
        cell_types=['L1', 'T2', 'Li15', 'MeLo10', 'L2'],
        target_mask=torch.tensor([False, False, False, False, True]),
        release_cap=.1, voltage_scale=.01)
    assert len(net.graded_edges) == 3
    zero = torch.zeros(1, 5)
    for _ in range(128):
        net.step(zero)
    net.enable_graded()
    net.release_history[(net.step_index-1) % net.history_length, 0] = .1
    activity = net.step(zero, capture_increments=True)
    assert torch.isclose(net.last_graded_impulse[0, 2],
                         torch.tensor(.1*transmitter_sign))
    assert torch.isclose(activity.feedforward_arrivals[0, 2],
                         torch.tensor(.1*transmitter_sign))
    assert activity.feedforward_arrivals[0, 4] == 0
    assert net.last_graded_impulse[0, 3] == 0
    predictive_impulse = (.1*net.excitatory_gain if transmitter_sign > 0
                          else -.1)
    assert torch.isclose(net.last_graded_impulse[0, 4],
                         torch.tensor(predictive_impulse))
    assert net.feedforward_current[0, 2]*transmitter_sign > 0
    assert net.predictive_current[0, 4]*transmitter_sign > 0
    net.step(zero)
    assert torch.isclose(net.last_graded_impulse[0, 3],
                         torch.tensor(predictive_impulse))
    assert net.predictive_current[0, 3]*transmitter_sign > 0


def test_tm9_current_residual_releases_on_its_existing_edge():
    graph = Graph(np.arange(1, 5), np.array([0, 1]),
                  np.array([3, 2]), np.ones(2, dtype=np.int64),
                  np.ones(4, dtype=np.int64), 1.)
    net = GradedVisualNetwork(graph, [1, 1], ['predictive', 'feedforward'],
        config=NeuronConfig(tau_sensory=.02, threshold=100.),
        cell_types=['L1', 'Tm9', 'T5d', 'L2'],
        target_mask=torch.tensor([False, False, False, True]),
        release_cap=.1, voltage_scale=1.,
        source_type='Tm9', source_state='current')
    zero = torch.zeros(1, 4)
    for _ in range(128):
        net.step(zero)
    net.enable_graded()
    net.feedforward_current[0, 1] = 1.
    net.step(zero)
    assert net.release_history[(net.step_index-1) % net.history_length, 0] == .1
    activity = net.step(zero, capture_increments=True)
    assert torch.isclose(net.last_graded_impulse[0, 2], torch.tensor(.1))
    assert torch.isclose(activity.feedforward_arrivals[0, 2], torch.tensor(.1))
