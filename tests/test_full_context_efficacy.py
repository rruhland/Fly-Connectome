import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from full_context_efficacy import MultiContextEfficacyNetwork
from signed_kinetics import AreaMatchedKineticsNetwork
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph


def networks():
    graph = Graph.from_contacts([10, 20, 30, 40], [10, 10, 40], [20, 30, 20],
                                [1, 1, 1], [1, 1, 1, -1], .5)
    args = (graph, [1, 1, 1], ['predictive']*3)
    config = NeuronConfig(tau_sensory=.02)
    baseline = AreaMatchedKineticsNetwork(*args, config=config)
    mask = torch.tensor([False, True, True, False])
    contextual = MultiContextEfficacyNetwork(*args, config=config, target_mask=mask)
    return baseline, contextual


def test_multiple_local_targets_equal_component_neural_parity():
    baseline, contextual = networks()
    assert contextual.targets.tolist() == [1, 2]
    assert contextual.incoming.tolist() == [0, 1, 2]
    for tick in range(40):
        if tick % 3 == 0:
            for net in (baseline, contextual):
                net.history[(net.step_index-1) % net.history_length, 0, 0] = True
        if tick % 4 == 0:
            for net in (baseline, contextual):
                net.history[(net.step_index-1) % net.history_length, 0, 3] = True
        sensory = torch.zeros(1, 4)
        if tick % 5 == 0:
            sensory[0, 1] = 30 if tick % 2 else -30
            sensory[0, 2] = -sensory[0, 1]
        old = baseline.step(sensory, capture_increments=True)
        new = contextual.step(sensory, capture_increments=True)
        for field in ('spikes', 'observed', 'predicted', 'arrival_edges', 'feedforward_arrivals'):
            torch.testing.assert_close(getattr(new, field), getattr(old, field), rtol=0, atol=0)
        for field in ('voltage', 'sensory_state', 'predictive_current',
                      'excitatory_prediction', 'inhibitory_prediction', 'adaptation'):
            torch.testing.assert_close(getattr(contextual, field), getattr(baseline, field), rtol=0, atol=0)


def test_each_target_selects_its_own_context_for_physical_current():
    _, net = networks()
    net.components[0] = torch.tensor([2., 3., 4.])
    net.components[1] = torch.tensor([6., 7., 1.])
    net.history[(net.step_index-1) % net.history_length, 0, 0] = True
    net.history[(net.step_index-1) % net.history_length, 0, 3] = True
    sensory = torch.zeros(1, 4); sensory[0, 1] = 30; sensory[0, 2] = -30
    first = net.step(sensory)
    assert net.current_context.tolist() == [[True, False]]
    torch.testing.assert_close(first.predicted[0, 1], torch.tensor(6*net.excitatory_gain-1))
    torch.testing.assert_close(first.predicted[0, 2], torch.tensor(3*net.excitatory_gain))
    sensory[0, 1] = -60; sensory[0, 2] = 60
    second = net.step(sensory)
    assert net.current_context.tolist() == [[False, True]]
    torch.testing.assert_close(second.predicted[0, 1],
                               torch.tensor(2*net.excitatory_gain*net.excitatory_decay-4*net.inhibitory_decay))
    torch.testing.assert_close(second.predicted[0, 2],
                               torch.tensor(7*net.excitatory_gain*net.excitatory_decay))
