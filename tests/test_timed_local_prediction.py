import sys
from pathlib import Path
from dataclasses import replace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from timed_local_prediction import TimedLocalPrediction
from fly_connectome.dynamics import Activity, Network, NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.plasticity import LearningConfig


def setup():
    graph = Graph.from_contacts([10, 20, 30], [10, 10], [20, 30], [1, 1], [1]*3, .5)
    net = Network(graph, [1, 1], ['predictive', 'predictive'], config=NeuronConfig(tau_sensory=.02))
    cfg = LearningConfig(prediction_encoding='signed-current-v1',
        visual_target='input-arrivals-v1', visual_eligibility='forecast-causal-v1',
        eta_prediction=.1, eta_reward=0., homeostasis_rate=0.)
    return net, TimedLocalPrediction(net, cfg, target=1,
        sensory_mask=torch.tensor([False, True, False]), sensory_gain=30.)


def activity(event=0., prediction=.4, arrived=True):
    z = torch.zeros(1, 3)
    sensory = z.clone(); sensory[0, 1] = event*30
    predicted = z.clone(); predicted[0, 1] = prediction
    env = torch.tensor([0], dtype=torch.long) if arrived else torch.empty(0, dtype=torch.long)
    edges = torch.tensor([0], dtype=torch.long) if arrived else torch.empty(0, dtype=torch.long)
    return Activity(torch.zeros(1, 3, dtype=torch.bool), sensory, predicted, env, edges,
                    feedforward_arrivals=z, sensory_input=sensory)


def advance(rule, ticks, event=0., prediction=.4):
    for i in range(ticks):
        rule.observe(activity(event if i == ticks-1 else 0., prediction, arrived=i == ticks-1), torch.zeros(1))


def test_surprise_acquires_local_reference_without_amplitude_credit():
    net, rule = setup()
    net.sensory_state[0, 1] = -30
    rule.observe(activity(-1.), torch.zeros(1))
    assert not rule.last_issue['gate']
    assert rule.reference == 0
    assert rule.event_count == 2
    assert rule.forecast[1].abs().sum() == 0
    advance(rule, 7)
    net.sensory_state[0, 1] = 26
    rule.observe(activity(1.), torch.zeros(1))
    assert rule.last_confirmation['gate'] is False
    assert rule.last_confirmation['delta'].abs().sum() == 0
    assert rule.reference == 30
    assert rule.last_issue['gate']
    assert rule.last_issue['prediction'] == pytest.approx(.4)
    assert rule.event_count == 3


def test_event_gain_uses_prior_counts_and_target_only_updates():
    net, rule = setup()
    rule.reference = 20.
    rule.quiet_count = 8
    rule.event_count = 1
    net.sensory_state[0, 1] = 20
    rule.observe(activity(), torch.zeros(1))
    assert rule.last_issue['gate']
    prior_eligibility = rule.forecast[1].clone()
    prior_weights = net.magnitudes.clone()
    advance(rule, 7)
    net.sensory_state[0, 1] = 20
    rule.observe(activity(1.), torch.zeros(1))
    expected = .1*8*(1-.4)*prior_eligibility
    assert rule.last_confirmation['gain'] == 8
    torch.testing.assert_close(rule.last_confirmation['delta'], expected, rtol=1e-6, atol=1e-6)
    assert rule.event_count == 2
    rule.synchronize()
    torch.testing.assert_close(net.magnitudes[0], prior_weights[0]+expected[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(net.magnitudes[1], prior_weights[1], rtol=0, atol=0)


def test_closed_gate_does_not_update_when_old_prediction_was_ungated():
    net, rule = setup()
    net.sensory_state[0, 1] = 3
    rule.observe(activity(prediction=.9), torch.zeros(1))
    advance(rule, 7)
    net.sensory_state[0, 1] = 4
    before = net.magnitudes.clone()
    rule.observe(activity(1.), torch.zeros(1))
    rule.synchronize()
    torch.testing.assert_close(net.magnitudes, before, rtol=0, atol=0)


def test_large_local_update_preserves_magnitude_bound_and_non_target_edge():
    net, rule = setup()
    rule.config = replace(rule.config, eta_prediction=100.)
    rule.reference = 20.
    net.sensory_state[0, 1] = 20
    rule.observe(activity(prediction=0.), torch.zeros(1))
    advance(rule, 7, prediction=0.)
    net.sensory_state[0, 1] = 20
    other = net.magnitudes[1].clone()
    rule.observe(activity(1., prediction=0.), torch.zeros(1))
    rule.synchronize()
    assert net.magnitudes[0] == 10
    torch.testing.assert_close(net.magnitudes[1], other, rtol=0, atol=0)
