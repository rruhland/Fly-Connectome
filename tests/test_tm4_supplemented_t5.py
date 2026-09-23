import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from tm4_supplemented_t5 import Tm4SupplementedT5Network, silent_source_release


def network(sign, *, tm4_tau=.250, tm4_cap=.1, tm4_scale=.02):
    graph = Graph(np.arange(1, 6), np.array([0, 1, 2]),
                  np.array([3, 4, 4]), np.ones(3, dtype=np.int64),
                  np.array([1, sign, 1, 1, 1]), 1.)
    net = Tm4SupplementedT5Network(graph, [1, 2, 1],
        ['predictive', 'feedforward', 'feedforward'],
        config=NeuronConfig(tau_sensory=.02, threshold=100.),
        cell_types=['L1', 'Tm4', 'Tm9', 'L2', 'T5d'],
        target_mask=torch.tensor([False, False, False, True, False]),
        release_cap=.1, voltage_scale=.02,
        source_type='Tm9', source_state='current',
        gate_gain=1., gate_cap=1., tm4_release_cap=tm4_cap,
        tm4_current_scale=tm4_scale, tm4_release_tau=tm4_tau)
    zero = torch.zeros(1, 5)
    for _ in range(128):
        net.step(zero)
    net.enable_graded()
    for _ in range(128):
        net.step(zero)
    net.enable_tm4()
    return net, zero


def test_tm4_release_uses_current_only_on_source_nonspike_ticks():
    current = torch.tensor([.08, .08, -.02])
    spikes = torch.tensor([False, True, False])
    actual = silent_source_release(current, torch.zeros(3),
        torch.zeros(3), spikes, scale=.02, cap=.1)
    assert torch.equal(actual, torch.tensor([.1, 0., 0.]))


@pytest.mark.parametrize('sign', [-1, 1])
def test_tm4_supplement_respects_measured_sign_delay_and_local_arm(sign):
    net, zero = network(sign)
    assert len(net.tm4_supplement_edges) == 1
    tick = net.step_index
    net.tm4_release_history[(tick-1) % net.history_length, 0] = .1
    net.step(zero)
    assert net.last_tm4_impulse[0, 4] == 0
    net.step(zero)
    assert torch.isclose(net.last_tm4_impulse[0, 4], torch.tensor(.1*sign))
    assert net.feedforward_current[0, 4]*sign > 0
    assert net.arm_traces[0, 0]*sign > 0


def test_shorter_tm4_release_baseline_suppresses_repeated_tonic_current():
    slow, zero = network(1, tm4_tau=.250, tm4_cap=1., tm4_scale=1.)
    fast, _ = network(1, tm4_tau=.050, tm4_cap=1., tm4_scale=1.)
    for net in (slow, fast):
        net.tm4_baseline.zero_()
        net.tm4_floor.zero_()
        net.feedforward_current[0, 1] = .1
        net.step(zero)
        net.feedforward_current[0, 1] = .1
        net.step(zero)
    slow_release = slow.tm4_release_history[
        (slow.step_index-1) % slow.history_length, 0]
    fast_release = fast.tm4_release_history[
        (fast.step_index-1) % fast.history_length, 0]
    assert fast_release < slow_release
