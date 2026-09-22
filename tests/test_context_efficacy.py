import sys
import math
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from context_efficacy import ContextEfficacyNetwork, ContextTimedPrediction
from signed_kinetics import AreaMatchedKineticsNetwork
from timed_local_prediction import TimedLocalPrediction
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.plasticity import LearningConfig


def networks():
    graph = Graph.from_contacts([10, 20, 30], [10, 30], [20, 20], [1, 1], [1, 1, -1], .5)
    config = NeuronConfig(tau_sensory=.02)
    args = (graph, [1, 1], ['predictive', 'predictive'])
    baseline = AreaMatchedKineticsNetwork(*args, config=config)
    contextual = ContextEfficacyNetwork(*args, config=config, target=1)
    return baseline, contextual


def test_equal_components_preserve_neural_dynamics_and_current():
    baseline, contextual = networks()
    cfg = LearningConfig(prediction_encoding='signed-current-v1',
                         visual_target='input-arrivals-v1',
                         visual_eligibility='forecast-causal-v1',
                         eta_prediction=0., eta_reward=0., homeostasis_rate=0.)
    args = dict(target=1, sensory_mask=torch.tensor([False, True, False]), sensory_gain=30.)
    old_rule = TimedLocalPrediction(baseline, cfg, **args)
    new_rule = ContextTimedPrediction(contextual, cfg, **args)
    for tick in range(32):
        if tick % 3 == 0:
            baseline.history[(baseline.step_index-1) % baseline.history_length, 0, 0] = True
            contextual.history[(contextual.step_index-1) % contextual.history_length, 0, 0] = True
        if tick % 4 == 0:
            baseline.history[(baseline.step_index-1) % baseline.history_length, 0, 2] = True
            contextual.history[(contextual.step_index-1) % contextual.history_length, 0, 2] = True
        sensory = torch.zeros(1, 3)
        sensory[0, 1] = (1 if tick % 6 < 3 else -1)*30 if tick % 4 == 0 else 0
        old = baseline.step(sensory, capture_increments=True)
        new = contextual.step(sensory, capture_increments=True)
        for name in ('spikes', 'observed', 'predicted', 'arrival_edges', 'feedforward_arrivals'):
            torch.testing.assert_close(getattr(new, name), getattr(old, name), rtol=0, atol=0)
        for name in ('voltage', 'sensory_state', 'predictive_current',
                     'excitatory_prediction', 'inhibitory_prediction', 'adaptation'):
            torch.testing.assert_close(getattr(contextual, name), getattr(baseline, name), rtol=0, atol=0)
        old_rule.observe(old, torch.zeros(1))
        new_rule.observe(new, torch.zeros(1))
        torch.testing.assert_close(new_rule.keys, old_rule.keys, rtol=0, atol=0)
        torch.testing.assert_close(new_rule.values, old_rule.values, rtol=0, atol=0)
        torch.testing.assert_close(new_rule.expected, old_rule.expected, rtol=0, atol=0)
        if tick % 8 == 0:
            for index in range(3):
                torch.testing.assert_close(new_rule.forecast[index], old_rule.forecast[index], rtol=0, atol=0)
            assert new_rule.last_issue['context'] == int(contextual.sensory_state[0, 1] > 0)


def test_local_context_selects_physical_current_on_same_fixed_edges():
    _, net = networks()
    assert net.incoming.tolist() == [0, 1]
    assert net.signs[net.incoming].tolist() == [1., -1.]
    net.components[0] = torch.tensor([2., 4.])
    net.components[1] = torch.tensor([6., 1.])
    net.history[(net.step_index-1) % net.history_length, 0, 0] = True
    net.history[(net.step_index-1) % net.history_length, 0, 2] = True
    sensory = torch.zeros(1, 3)
    sensory[0, 1] = 30
    first = net.step(sensory)
    assert net.current_context == 1
    expected = 6*net.excitatory_gain-1
    torch.testing.assert_close(first.predicted[0, 1], torch.tensor(expected))
    sensory[0, 1] = -60
    second = net.step(sensory)
    assert net.current_context == 0
    expected = 2*net.excitatory_gain*net.excitatory_decay-4*net.inhibitory_decay
    torch.testing.assert_close(second.predicted[0, 1], torch.tensor(expected))
    assert net.components.min() >= 0
    assert net.components.max() <= 10


def test_only_issued_context_receives_local_credit_and_bounds():
    _, net = networks()
    config = LearningConfig(prediction_encoding='signed-current-v1',
                            visual_target='input-arrivals-v1',
                            visual_eligibility='forecast-causal-v1',
                            eta_prediction=100., eta_reward=0., homeostasis_rate=0.)
    rule = ContextTimedPrediction(net, config, target=1,
                                  sensory_mask=torch.tensor([False, True, False]), sensory_gain=30.)
    rule.reference = 20.
    net.sensory_state[0, 1] = 20/math.exp(-net.config.dt/net.config.tau_sensory)
    initial = net.components.clone()
    for tick in range(9):
        sensory = torch.zeros(1, 3)
        if tick == 0:
            net.history[(net.step_index-1) % net.history_length, 0, 0] = True
        if tick == 8:
            sensory[0, 1] = -30
        activity = net.step(sensory, capture_increments=True)
        rule.observe(activity, torch.zeros(1))
        if tick == 0:
            assert rule.last_issue['context'] == 1
            assert rule.last_issue['gate']
            assert rule.last_issue['edges'].tolist() == [0]
    assert rule.last_confirmation['context'] == 1
    assert rule.last_confirmation['delta'].numel() == 1
    rule.synchronize()
    torch.testing.assert_close(net.components[0], initial[0], rtol=0, atol=0)
    torch.testing.assert_close(net.components[1, 1], initial[1, 1], rtol=0, atol=0)
    torch.testing.assert_close(net.components[1, 0],
                               torch.clamp(initial[1, 0]+rule.last_confirmation['delta'][0], 0, 10))
    assert 0 <= float(net.components.min()) <= float(net.components.max()) <= 10


def test_closed_gate_surprise_updates_reference_without_efficacy_credit():
    _, net = networks()
    cfg = LearningConfig(prediction_encoding='signed-current-v1',
                         visual_target='input-arrivals-v1',
                         visual_eligibility='forecast-causal-v1',
                         eta_prediction=1., eta_reward=0., homeostasis_rate=0.)
    rule = ContextTimedPrediction(net, cfg, target=1,
                                  sensory_mask=torch.tensor([False, True, False]), sensory_gain=30.)
    initial = net.components.clone()
    net.history[(net.step_index-1) % net.history_length, 0, 0] = True
    for tick in range(9):
        sensory = torch.zeros(1, 3)
        if tick == 0:
            sensory[0, 1] = -30
        elif tick == 8:
            sensory[0, 1] = 30
        activity = net.step(sensory, capture_increments=True)
        rule.observe(activity, torch.zeros(1))
        if tick == 0:
            assert not rule.last_issue['gate']
    assert rule.last_confirmation['gain'] >= 1
    assert rule.last_confirmation['delta'].abs().sum() == 0
    assert rule.reference > 0
    rule.synchronize()
    torch.testing.assert_close(net.components, initial, rtol=0, atol=0)
