import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from t5_local_learning import T5FramePrediction, T5BalancedFramePrediction
from fly_connectome.dynamics import Activity, Network, NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.plasticity import LearningConfig


def test_t5_frame_rule_captures_only_local_target_edges():
    graph = Graph(np.array([1, 2, 3]), np.array([0, 0]),
                  np.array([1, 2]), np.array([2, 3]),
                  np.ones(3, dtype=np.int64), .1)
    net = Network(graph, [1, 1], ['predictive', 'predictive'],
        config=NeuronConfig(tau_sensory=.02))
    cfg = LearningConfig(visual_eligibility='forecast-causal-v1')
    rule = T5FramePrediction(net, cfg,
        target_mask=torch.tensor([False, True, False]))
    rule.keys = torch.tensor([0, 1])
    rule.values = torch.tensor([.5, .5])
    rule._capture_forecast()
    assert torch.equal(rule.forecast[0], torch.tensor([0]))
    rule._accumulate(torch.tensor([0, 1]), torch.tensor([1., 1.]))
    assert torch.equal(rule.proposals, torch.zeros(2))


def test_balanced_t5_rule_scales_only_confirmed_local_event_credit():
    graph = Graph(np.array([1, 2, 3]), np.array([0, 0]),
                  np.array([1, 2]), np.array([2, 3]),
                  np.ones(3, dtype=np.int64), .1)
    net = Network(graph, [1, 1], ['predictive', 'predictive'],
        config=NeuronConfig(tau_sensory=.02))
    cfg = LearningConfig(visual_target='input-arrivals-v1',
        visual_eligibility='forecast-causal-v1', eta_prediction=.01)
    rule = T5BalancedFramePrediction(net, cfg,
        target_mask=torch.tensor([False, True, False]),
        sensory_mask=torch.zeros(3, dtype=torch.bool))
    rule.tick = 8
    rule.forecast = (torch.tensor([0]), torch.tensor([.5]), torch.tensor([0.]))
    rule.event_count[1] = 1.
    rule.quiet_count[1] = 9.
    zeros = torch.zeros(1, 3)
    activity = Activity(torch.zeros(1, 3, dtype=torch.bool), zeros, zeros,
        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
        torch.tensor([[0., .5, 0.]]), zeros)
    rule.observe(activity, torch.zeros(1))
    assert torch.isclose(rule.proposals[0], torch.tensor(.01))
    assert rule.last_event_gain == 4.
    assert rule.proposals[1] == 0
    assert torch.isclose(rule.event_count[1], torch.tensor(1.98))
    assert torch.isclose(rule.quiet_count[1], torch.tensor(8.82))
