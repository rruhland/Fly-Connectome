import sys
from pathlib import Path
import math

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from full_context_efficacy import MultiContextEfficacyNetwork, MultiContextTimedPrediction
from context_efficacy import ContextEfficacyNetwork, ContextTimedPrediction
from signed_kinetics import AreaMatchedKineticsNetwork
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.plasticity import LearningConfig


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


def rule_setup(eta=1.):
    _, net = networks()
    config = LearningConfig(prediction_encoding='signed-current-v1',
        visual_target='input-arrivals-v1', visual_eligibility='forecast-causal-v1',
        eta_prediction=eta, eta_reward=0., homeostasis_rate=0.)
    rule = MultiContextTimedPrediction(net, config,
        sensory_mask=torch.tensor([False, True, True, False]), sensory_gain=30.)
    return net, rule


def test_per_target_issue_context_routes_only_local_component_credit():
    net, rule = rule_setup(eta=100.)
    rule.reference.fill_(20)
    decay = math.exp(-net.config.dt/net.config.tau_sensory)
    net.sensory_state[0, 1] = 20/decay
    net.sensory_state[0, 2] = -20/decay
    initial = net.components.clone()
    net.history[(net.step_index-1) % net.history_length, 0, 0] = True
    net.history[(net.step_index-1) % net.history_length, 0, 3] = True
    for tick in range(9):
        sensory = torch.zeros(1, 4)
        if tick == 8:
            sensory[0, 1] = -30
            sensory[0, 2] = 30
        activity = net.step(sensory, capture_increments=True)
        rule.observe(activity, torch.zeros(1))
        if tick == 0:
            assert rule.last_issue['context'].tolist() == [True, False]
            assert rule.last_issue['gate'].tolist() == [True, True]
            assert rule.last_issue['edges'].tolist() == [0, 1, 2]
    assert rule.last_confirmation['context'].tolist() == [True, False, True]
    edges = rule.last_confirmation['edges']
    delta = rule.last_confirmation['delta']
    expected = initial.clone()
    for edge, change, context in zip(edges.tolist(), delta.tolist(),
                                     rule.last_confirmation['context'].tolist()):
        expected[int(context), net.incoming_lookup[edge]] += change
    expected.clamp_(0, 10)
    rule.synchronize()
    torch.testing.assert_close(net.components, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(net.magnitudes, expected[0], rtol=1e-6, atol=1e-6)
    assert 0 <= float(net.components.min()) <= float(net.components.max()) <= 10


def test_closed_gate_surprise_updates_only_local_timing_reference():
    net, rule = rule_setup()
    initial = net.components.clone()
    net.history[(net.step_index-1) % net.history_length, 0, 0] = True
    for tick in range(9):
        sensory = torch.zeros(1, 4)
        if tick == 0:
            sensory[0, 1] = -30
        elif tick == 8:
            sensory[0, 1] = 30
        activity = net.step(sensory, capture_increments=True)
        rule.observe(activity, torch.zeros(1))
        if tick == 0:
            assert rule.last_issue['gate'].tolist() == [False, False]
    assert rule.last_confirmation['delta'].abs().sum() == 0
    assert rule.reference[0] > 0 and rule.reference[1] == 0
    assert rule.event_count.tolist() == [3., 1.]
    rule.synchronize()
    torch.testing.assert_close(net.components, initial, rtol=0, atol=0)


def test_one_target_rule_matches_verified_single_target_rule_tick_by_tick():
    graph = Graph.from_contacts([10, 20, 30], [10, 30], [20, 20], [1, 1], [1, 1, -1], .5)
    args = (graph, [1, 1], ['predictive', 'predictive'])
    config = NeuronConfig(tau_sensory=.02)
    old = ContextEfficacyNetwork(*args, config=config, target=1)
    new = MultiContextEfficacyNetwork(*args, config=config,
                                      target_mask=torch.tensor([False, True, False]))
    learning = LearningConfig(prediction_encoding='signed-current-v1',
        visual_target='input-arrivals-v1', visual_eligibility='forecast-causal-v1',
        eta_prediction=.1, eta_reward=0., homeostasis_rate=0.)
    kwargs = dict(sensory_mask=torch.tensor([False, True, False]), sensory_gain=30.)
    old_rule = ContextTimedPrediction(old, learning, target=1, **kwargs)
    new_rule = MultiContextTimedPrediction(new, learning, **kwargs)
    for tick in range(40):
        if tick % 4 == 0:
            for net in (old, new):
                net.history[(net.step_index-1) % net.history_length, 0, 0] = True
        if tick % 6 == 0:
            for net in (old, new):
                net.history[(net.step_index-1) % net.history_length, 0, 2] = True
        sensory = torch.zeros(1, 3)
        if tick % 8 == 0 and tick:
            sensory[0, 1] = 30 if tick % 16 else -30
        old_activity = old.step(sensory, capture_increments=True)
        new_activity = new.step(sensory, capture_increments=True)
        torch.testing.assert_close(new_activity.predicted, old_activity.predicted, rtol=0, atol=0)
        old_rule.observe(old_activity, torch.zeros(1))
        new_rule.observe(new_activity, torch.zeros(1))
        torch.testing.assert_close(new_rule.keys, old_rule.keys, rtol=0, atol=0)
        torch.testing.assert_close(new_rule.values, old_rule.values, rtol=0, atol=0)
        torch.testing.assert_close(new_rule.expected, old_rule.expected, rtol=0, atol=0)
        if tick % 8 == 0:
            assert bool(new_rule.last_issue['gate'][0]) == old_rule.last_issue['gate']
            assert bool(new_rule.last_issue['context'][0]) == bool(old_rule.last_issue['context'])
            torch.testing.assert_close(new_rule.forecast[1], old_rule.forecast[1], rtol=0, atol=0)
            torch.testing.assert_close(new_rule.forecast[2], old_rule.forecast[2], rtol=0, atol=0)
        if tick % 8 == 7:
            old_rule.synchronize()
            new_rule.synchronize()
            torch.testing.assert_close(new.components, old.components, rtol=0, atol=0)
